import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  

class StraightThroughSoftmax(nn.Module):
    def __init__(self):
        super(StraightThroughSoftmax, self).__init__()

    # def forward(self, logits):
    #     probs = F.softmax(logits, dim=-1)
    #     print(probs.shape)
    #     sampled_index = torch.multinomial(probs, 1)
    #     one_hot = torch.zeros_like(probs).scatter_(1, sampled_index, 1)
    #     return (one_hot - probs).detach() + probs, probs

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)
        
        if self.training:
            sampled_index = torch.multinomial(probs, 1)
            one_hot = torch.zeros_like(probs).scatter_(1, sampled_index, 1)
            return (one_hot - probs).detach() + probs, probs
        else:
            return probs, probs

class CNN1dEstimator(nn.Module):
    """
    1D Convolutional Neural Network for processing sequential data.
    
    Attributes:
        activation_fn: Activation function used throughout the network
        tsteps: Fixed length of input sequences (temporal steps)
        encoder: Linear projection layer that expands feature dimensions
        conv_layers: 1D convolutional blocks configured for specific sequence lengths
        linear_output: Final processing layer producing the output
        
    Args:
        activation_fn: Activation function (e.g., nn.ReLU)
        input_size: Number of features per timestep
        tsteps: Number of timesteps in input sequences (must be 10, 20, or 50)
        output_size: Dimensionality of the output vector
        tanh_encoder_output: Unused in current implementation (retained for compatibility)
    """
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        """
        Initializes CNN1dEstimator with configurable architecture components.
        
        The network architecture dynamically adapts to different sequence lengths (tsteps):
        - For tsteps=50: Uses 3 convolutional layers with stride reduction
        - For tsteps=20: Uses 2 convolutional layers with stride reduction
        - For tsteps=10: Uses 2 convolutional layers with minimal reduction
        
        Raises:
            ValueError: If tsteps is not 10, 20, or 50
        """
        super(CNN1dEstimator, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )
        

    def forward(self, obs):
        """
        Forward pass processing:
        
        Input shape: (batch_size, tsteps, input_size)
        Output shape: (batch_size, output_size)
        
        Args:
            obs: Input tensor containing sequential data
        
        Returns:
            Processed output tensor
        """
        
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd*T, -1]))#projection = self.encoder(obs.reshape([nd, T])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dims, 
                 hidden_dims, 
                 latent_dim = 16,
                 history_len = 20,
                 feet_contact_dim = 6,
                 kl_w = 0.1,
                 prior_mu=None):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dims (list of int): List of hidden layer dimensions.
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.encoder = CNN1dEstimator(nn.ReLU(), int(input_dim//history_len), history_len, hidden_dims[-1])

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], output_dims-feet_contact_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        # Feet contact decoder
        FC_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            FC_decoder_layers.append(nn.Linear(prev_dim, h_dim))
            FC_decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        FC_decoder_layers.append(nn.Linear(hidden_dims[0], feet_contact_dim))
        FC_decoder_layers.append(nn.Sigmoid())
        self.FC_decoder = nn.Sequential(*FC_decoder_layers)
        self.prior_mu = prior_mu
        self.kl_w = kl_w

    def encode(self, x):
        """
        Encode input into latent distribution parameters.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        
        Returns:
            z (tensor): Sampled latent vector of shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_representation(self, x):
        """
        Get latent representation without reconstruction.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            z (tensor): Sampled latent vector of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        """
        Reconstruct outputs from latent vector.
        
        Args:
            z (tensor): Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            recon_x (tensor): Main reconstruction of shape (batch_size, output_dims)
            recon_contact (tensor): Foot contact probabilities of shape (batch_size, feet_contact_dim)
        """
        return self.decoder(z), self.FC_decoder(z)

    def forward(self, x):
        """
        Full VAE forward pass.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            recon_x (tensor): Main reconstruction of shape (batch_size, output_dims)
            recon_contact (tensor): Foot contact probabilities of shape (batch_size, feet_contact_dim)
            mu (tensor): Latent mean of shape (batch_size, latent_dim)
            logvar (tensor): Log variance of shape (batch_size, latent_dim)
        """
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, recon_contact= self.decode(z)
        return recon_x, recon_contact, mu, logvar

class DHA_VAE(nn.Module):
    """
    Dual Hierarchical Architecture with Variational Autoencoders (DHA-VAE)
    Combines mode recognition (DHA) with mode-specific dynamics modeling (TsDyn VAE modules)
    
    Parameters:
    -----------
    num_his_obs : int
        Dimension of historical observation data (input to TsDyn modules).
        Represents flattened history: history_len * obs_dim_per_step.
    
    num_recon : int
        Dimension of reconstruction targets (VAE output dimension).
        Should be next_obs_dim + contact_dim (concatenated targets).
    
    history_len : int
        Number of historical timesteps used for sequential modeling. Choose from [10,20,50]
    
    num_actor_obs : int
        Dimension of current agent observation (input to DHA module).
        For MDP, num_his_obs=num_actor_obs*history_len.
    
    num_modes : int
        Number of discrete behavioral modes (e.g., walking, running).
        Determines number of parallel TsDyn modules and DHA output dimension.
    
    tsdyn_hidden_dims : List[int], optional
        Hidden layer dimensions for VAE encoder/decoder networks. 
        Default: [256, 128, 64]
    
    tsdyn_latent_dims : int, optional
        Dimension of latent space for each mode-specific VAE. 
        Default: 32
    
    Architecture:
    -------------
    1. DHA (Dynamic Hierarchy Analyzer):
        - Input: current observation (num_actor_obs)
        - Output: mode probabilities (num_modes)
        - Uses StraightThroughSoftmax for differentiable discrete sampling
    
    2. TsDyn_modules (Mode-specific Dynamics):
        - Contains num_modes independent VAEs
        - Each VAE:
            - Input: historical observations (num_his_obs)
            - Output: reconstructed next_obs + contact (num_recon)
            - Hidden layers: tsdyn_hidden_dims
            - Latent space: tsdyn_latent_dims
    """
    def __init__(self,
                    num_his_obs,
                    num_recon,
                    history_len,
                    num_actor_obs,
                    num_modes=3,
                    tsdyn_hidden_dims= [256, 128, 64],
                    tsdyn_latent_dims=32,
                    feet_contact_dim = 6):
        super(DHA_VAE, self).__init__()
        self.DHA = nn.Sequential(
            nn.Linear(num_actor_obs, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, num_modes),
            StraightThroughSoftmax()
        )
        self.TsDyn_modules = nn.ModuleList() 
        self.history_len = history_len
        self.num_recon = num_recon
        TsDyn_input_dim = num_his_obs
        TsDyn_output_dim = self.num_recon
        for i in range (num_modes):
            self.TsDyn_modules.append(VAE(TsDyn_input_dim, \
                            TsDyn_output_dim, tsdyn_hidden_dims, tsdyn_latent_dims, \
                            self.history_len, kl_w=0.9, prior_mu=0,feet_contact_dim=6))
        self._initialize_weights()

    def get_representation(self,observations):
        '''
        observations: (batch_size,history_length,obs_dim)
        '''
        observations = observations.float().unsqueeze(0)
        mode_latent, prob = self.DHA(observations[:,-1,:])
        mode_latent = mode_latent.detach()
        representation_list = []
        # get transition dynamics representation
        for i, sub_net in enumerate(self.TsDyn_modules):
            representation_list.append(sub_net.get_representation(observations).unsqueeze(1))
        representation = torch.cat(representation_list, dim=1)
        #print(representation.shape,mode_latent.unsqueeze(1).shape)
        representation = torch.bmm(mode_latent.unsqueeze(1), representation).squeeze(1)
        #print(representation.shape)
        return representation

    def vae_loss(self, 
                obs: torch.Tensor, 
                next_obs: torch.Tensor,
                contact: torch.Tensor) -> torch.Tensor:
        """
        计算VAE相关损失
        输入:
            obs: [batch_size,history_length, obs_dim] 当前观测
            next_obs: [batch_size, obs_dim] 下一观测（重建目标）
            contact: [batch_size, contact_dim] 接触状态（重建目标）
        输出: 
            loss: 标量损失值
        """
        # 获取当前模式
        #print(obs.shape)
        # obs = obs.cpu()
        # next_obs = next_obs.cpu()
        # contact = contact.cpu()
        batch_size = obs.shape[0]
        mode_latent, _ = self.DHA(obs[:,-1,:])
        
        # 计算各VAE损失
        losses = []
        obs_losses = []
        contact_losses = []
        kl = []
        for i, vae in enumerate(self.TsDyn_modules):
            recon_obs, recon_contact, mu, logvar = vae(obs.flatten(1))
            #print(next_obs.shape,recon_obs.shape)
            # reconstruction loss
            obs_loss = F.mse_loss(recon_obs, next_obs[:,-1], reduction='none')
            contact_loss = F.binary_cross_entropy(recon_contact, contact[:,-1], reduction='none')
            
            # KL div
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # weighted_div(obs loss, contact loss, kl_div)
            #print(obs_loss.shape,contact_loss.shape,kl_div.shape)
            loss = obs_loss.sum(dim=-1) + 0.8 * contact_loss.sum(dim=-1) + 0.9 * kl_div
            losses.append(loss.unsqueeze(1))
            obs_losses.append(obs_loss.sum(dim=-1).unsqueeze(1))
            contact_losses.append(contact_loss.sum(dim=-1).unsqueeze(1))
            kl.append(kl_div.unsqueeze(1))
        
        # 模式加权
        total_loss = (torch.cat(losses, dim=1) * mode_latent).sum()/batch_size
        obs_losses,contact_losses,kl = (torch.cat(obs_losses, dim=1) * mode_latent).sum()/batch_size,(torch.cat(contact_losses, dim=1) * mode_latent).sum()/batch_size,(torch.cat(kl, dim=1) * mode_latent).sum()/batch_size
        return total_loss,obs_losses,contact_losses,kl

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

def test_dha_vae():
    # 配置参数
    batch_size = 32
    num_his_obs = 100    
    num_recon = 14      
    history_len = 10      
    num_actor_obs = 10   
    num_modes = 3        
    latent_dim = 32      
    
    # 创建模型
    model = DHA_VAE(
        num_his_obs=num_his_obs,
        num_recon=num_recon,
        history_len=history_len,
        num_actor_obs=num_actor_obs,
        num_modes=num_modes,
        tsdyn_latent_dims=latent_dim
    )
    
    
    # 创建测试数据
    obs = torch.randn(batch_size,history_len,num_actor_obs)           # 当前观测
    next_obs = torch.randn(batch_size,num_actor_obs)      # 下一状态 (10维)
    contact = torch.randint(0, 2, (batch_size, 4)).float() # 接触状态 (4维)
    

    print("Input shapes:")
    print(f"obs:        {obs.shape}")
    print(f"next_obs:   {next_obs.shape}")
    print(f"contact:    {contact.shape}\n")
    
    # 测试1: 获取表示向量
    representation = model.get_representation(obs)
    print("Test 1: Representation output")
    print(f"Shape: {representation.shape} (expected: [{batch_size}, {latent_dim}])\n")
    
    # 测试2: 计算VAE损失
    loss = model.vae_loss(obs, next_obs, contact)
    print("Test 2: VAE Loss")
    print(f"Loss value: {loss.item():.4f} (should be non-zero)")
    print(f"Requires grad: {loss.requires_grad}\n")
    
    # 测试3: 前向传播梯度检查
    print("Test 3: Gradient check")
    loss.backward()
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Model parameters have gradients: {has_gradients}")
    
    # 测试4: 模式概率输出
    with torch.no_grad():
        model.eval()  # 切换到推理模式
        mode_probs = model.DHA(obs)
        print("\nTest 4: Mode probabilities (inference mode)")
        print(f"Shape: {mode_probs[0].shape} (expected: [{batch_size}, {num_modes}])")
        print("Sample probabilities:")
        print(mode_probs[0][0].detach().numpy())
        
        model.train()  # 切换回训练模式
        hard_probs = model.DHA(obs)
        print("\nTest 5: Mode selection (training mode)")
        print("Sample one-hot like output:")
        print(hard_probs[0][0].detach().numpy())

# 运行测试
if __name__ == "__main__":
    test_dha_vae()