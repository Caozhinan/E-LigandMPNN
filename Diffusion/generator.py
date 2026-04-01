from typing import Any, Callable, Optional
import torch
from Diffusion.utils import centre_random_augmentation, inverse_centre_augmentation
import pdb

class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        #rnd_normal = torch.ones(size=size, device=device) + torch.ones(size=size, device=device) + torch.ones(size=size, device=device) + torch.ones(size=size, device=device) 
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        #noise_level = torch.clamp( (rnd_normal * self.p_std + self.p_mean), max=2.9 ).exp() * self.sigma_data
        # import pdb
        # pdb.set_trace()
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        xyz_37_valid = input_feature_dict["xyz_37_valid"]        # [B, N_atom, 3]
        noise_mask = ~(input_feature_dict["backbone_mask"].bool()) & input_feature_dict["atom_valid"].bool()  # [B, N]
        noise_mask = noise_mask.unsqueeze(-1).float()

        xyz_37_valid_expand = xyz_37_valid.unsqueeze(1).expand(-1, chunk_n_sample, -1, -1)     # [B, S, N, 3]
        noise_mask_expand = noise_mask.unsqueeze(1).expand(-1, chunk_n_sample, -1, -1)         # [B, S, N, 1]
        
        atom_name_index = input_feature_dict["atom37_index"]
        ca_mask = (atom_name_index == 1).float()      # [B, N_atom]
        ca_mask_expand = ca_mask.unsqueeze(-1)     # [B, N_atom, 1]

        # 用 scatter 加权求和取 CA 坐标（[B, N_res, 3]）
        B, N_atom = atom_to_token_idx.shape
        N_res = atom_to_token_idx.max().item() + 1
        device = xyz_37_valid.device

        ca_coords_per_res = torch.zeros(B, N_res, 3, device=device)
        count_per_res = torch.zeros(B, N_res, 1, device=device)

        ca_coords_per_res.scatter_add_(1,
            atom_to_token_idx.unsqueeze(-1).expand(-1, -1, 3),
            xyz_37_valid * ca_mask_expand
        )
        count_per_res.scatter_add_(1,
            atom_to_token_idx.unsqueeze(-1),
            ca_mask_expand
        )
        ca_coords_per_res = ca_coords_per_res / (count_per_res + 1e-8)   # [B, N_res, 3]

        # --- 2. 每个原子获取自己残基的 CA 坐标 ---
        ca_coords_per_atom = torch.gather(
            ca_coords_per_res,
            dim=1,
            index=atom_to_token_idx.unsqueeze(-1).expand(-1, -1, 3)   # [B, N_atom, 3]
        )  # [B, N_atom, 3]

        # --- 3. 扩展到采样维度 + 替代 ---
        ca_coords_expand = ca_coords_per_atom.unsqueeze(1).expand(-1, chunk_n_sample, -1, -1)  # [B, S, N, 3]
        x_l = xyz_37_valid_expand * (1.0 - noise_mask_expand) + ca_coords_expand * noise_mask_expand

        # origin
        noise = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3),
            device=device,
            dtype=dtype
        )  # [B, S, N, 3]

        # our
        # noise = torch.randn(
        #     size=(*batch_shape, chunk_n_sample, N_atom, 3),
        #     device=device,
        #     dtype=dtype
        # )  # [B, S, N, 3]

        #x_l = xyz_37_valid_expand * (1.0 - noise_mask_expand) + noise * noise_mask_expand
        #x_l = x_l * (1.0 - noise_mask_expand) + noise * noise_mask_expand
        x_l = x_l + noise * noise_mask_expand

        x_l = centre_random_augmentation(
            x_input_coords=x_l,
            N_sample=1,
            centre_only=True,
            mask=input_feature_dict["backbone_mask"], #input_feature_dict["atom_valid"],
        ).squeeze(dim=-3).to( dtype )

        #mm = inverse_centre_augmentation( x_l, noise_mask_expand, xyz_37_valid_expand, eps=0 )

        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
                                
            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_mask_expand * noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )
            with torch.no_grad():
                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    noise_mask=noise_mask,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta  #这个step_scale_eta可以做消融实验

        x_l = inverse_centre_augmentation( x_l, input_feature_dict["backbone_mask"].unsqueeze(0).unsqueeze(-1), xyz_37_valid_expand, eps=0 ) #1e-8 )

        # coord1 = x_l[0][0][~input_feature_dict["backbone_mask"].bool().squeeze(0)]
        # coord2 = xyz_37_valid_expand[0][0][~input_feature_dict["backbone_mask"].bool().squeeze(0)]
        # print(coord1[:10])
        # print(coord2[:10])
        # rmse = torch.sqrt(torch.mean((coord1 - coord2) ** 2))
        # print("RMSE:", rmse.item())
        # import pdb
        # pdb.set_trace()
        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l

def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    #label_dict: dict[str, Any],  #need coordinate and mask
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    # batch_size_shape = label_dict["coordinate"].shape[:-2]
    # device = label_dict["coordinate"].device
    # dtype = label_dict["coordinate"].dtype

    batch_size_shape = input_feature_dict["xyz_37_valid"].shape[:-2]
    device = input_feature_dict["xyz_37_valid"].device
    dtype = input_feature_dict["xyz_37_valid"].dtype

    # Areate N_sample versions of the input structure by randomly rotating and translating
    # x_gt_augment = centre_random_augmentation(
    #     x_input_coords=label_dict["coordinate"],
    #     N_sample=N_sample,
    #     mask=label_dict["coordinate_mask"],
    # ).to(
    #     dtype
    # )  # [..., N_sample, N_atom, 3]
    x_gt_augment = centre_random_augmentation(
        x_input_coords=input_feature_dict["xyz_37_valid"],
        N_sample=N_sample,
        mask=input_feature_dict["backbone_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]

    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype) #每份会取不同sampler数量的噪声
    # import pdb
    # pdb.set_trace()
    # noise: [..., N_sample, N_atom, 3]
    # import pdb
    # pdb.set_trace()
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]  #backbone不进行添加

    noise_mask = ~(input_feature_dict["backbone_mask"].bool()) & input_feature_dict["atom_valid"].bool() # [batch_size, N_atom]
    noise_mask = noise_mask.unsqueeze(-1).float()  # [batch_size, N_atom, 1]
    
    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise * noise_mask,  #噪声，修改了一下
            noise_mask=noise_mask,  #噪声mask
            t_hat_noise_level=sigma,  #噪声水平
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise * noise_mask)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]  #哪些sample加噪声
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                noise_mask=noise_mask,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
            ) #去噪网络,s原始的特征，z是pair吧
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)
    
    # print(sigma)
    # print( "x_denoised", x_denoised[0][0][:10] )
    # print( "x_noisy", x_noisy_i[0][0][:10])
    # import pdb
    # pdb.set_trace()
    return x_gt_augment, x_denoised, sigma
