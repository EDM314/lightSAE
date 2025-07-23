import torch
import torch.nn as nn
import torch.nn.functional as F


class HeEmb(nn.Module):
    emb_type_list=[
        'SharedEmb',

        'IndEmb',
        'SharedIndEmb',

        'IndLoRAEmb',
        'SharedIndLoRAEmb',

        'MoEEmb',
        'SharedMoEEmb',
        
        'MoELoRAEmb',
        'SharedMoELoRAEmb',

        ] 
    def __init__(self, n, input_dim, output_dim, model_type, rank=10, num_experts=10, moe_router_type='learned', moe_mlp_hidden_dim=64, channel_identity_dim=32, use_softmax=True, grouped_bias=True, force_scheme=None):
        super(HeEmb, self).__init__()
        
        parts = model_type.split('_')
        self.emb_type = parts[0] if len(parts) > 0 else 'SharedEmb'
        self.bias_type_1 = parts[1] if len(parts) > 1 else 'BiasOn'

        bias = False if self.bias_type_1 == "BiasOff" else True

        assert self.emb_type in self.__class__.emb_type_list
        
        self.rank = rank
        self.num_experts = num_experts
        self.moe_router_type = moe_router_type
        self.moe_mlp_hidden_dim = moe_mlp_hidden_dim
        self.channel_identity_dim = channel_identity_dim
        self.use_softmax = use_softmax
        self.grouped_bias = grouped_bias
        self.force_scheme = force_scheme

        if self.emb_type == 'SharedEmb':
            self.weight = nn.Parameter(torch.empty(input_dim,output_dim))
            self.bias = nn.Parameter(torch.empty(output_dim)) if bias else None
            nn.init.normal_(self.weight, mean=0.0, std=1e-2)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.emb_type == 'IndEmb':
            self.weight = nn.Parameter(torch.empty(n,input_dim,output_dim))
            self.bias = nn.Parameter(torch.empty(n,output_dim)) if bias else None
            nn.init.normal_(self.weight, mean=0.0, std=1e-2)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        elif self.emb_type == 'SharedIndEmb':
            self.shared_weight = nn.Parameter(torch.empty(input_dim,output_dim))
            self.ind_weight = nn.Parameter(torch.empty(n,input_dim,output_dim))
            self.bias = nn.Parameter(torch.empty(n,output_dim)) if bias else None
            nn.init.normal_(self.shared_weight, mean=0.0, std=1e-2)
            nn.init.normal_(self.ind_weight, mean=0.0, std=1e-5)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        elif self.emb_type == 'SharedIndLoRAEmb':
            assert self.rank is not None
            self.shared_weight = nn.Parameter(torch.empty(input_dim, output_dim)) 
            self.lora_A = nn.Parameter(torch.empty(n, input_dim, rank))
            self.lora_B = nn.Parameter(torch.empty(n, rank, output_dim))
            self.bias = nn.Parameter(torch.empty(n,output_dim)) if bias else None
            nn.init.normal_(self.shared_weight, mean=0.0, std=1e-2)
            nn.init.normal_(self.lora_A, mean=0.0, std=1e-5)
            nn.init.zeros_(self.lora_B)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        elif self.emb_type == 'IndLoRAEmb':
            assert self.rank is not None
            self.lora_A = nn.Parameter(torch.empty(n, input_dim, self.rank))
            self.lora_B = nn.Parameter(torch.empty(n, self.rank, output_dim))
            self.bias = nn.Parameter(torch.empty(n, output_dim)) if bias else None
            nn.init.normal_(self.lora_A, mean=0.0, std=1e-2)
            nn.init.zeros_(self.lora_B)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        elif self.emb_type == 'MoEEmb':
            self._init_moe_router(n, input_dim_for_dynamic_mlp=input_dim)

            self.experts = nn.Parameter(torch.empty(self.num_experts, input_dim, output_dim))
            nn.init.normal_(self.experts, mean=0.0, std=1e-2)

            if bias:
                if self.grouped_bias:
                    self.expert_biases = nn.Parameter(torch.empty(self.num_experts, output_dim))
                    nn.init.zeros_(self.expert_biases)
                    self.bias = None
                else:
                    self.expert_biases = None
                    self.bias = nn.Parameter(torch.empty(output_dim))
                    nn.init.zeros_(self.bias)
            else:
                self.expert_biases = None
                self.bias = None
        elif self.emb_type == 'SharedMoEEmb':
            self._init_moe_router(n, input_dim_for_dynamic_mlp=input_dim)

            self.shared_weight = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.normal_(self.shared_weight, mean=0.0, std=1e-2)
            self.experts = nn.Parameter(torch.empty(self.num_experts, input_dim, output_dim))
            nn.init.normal_(self.experts, mean=0.0, std=1e-5)

            if bias:
                if self.grouped_bias:
                    self.expert_biases = nn.Parameter(torch.empty(self.num_experts, output_dim))
                    nn.init.zeros_(self.expert_biases)
                    self.bias = None
                else:
                    self.expert_biases = None
                    self.bias = nn.Parameter(torch.empty(output_dim))
                    nn.init.zeros_(self.bias)
            else:
                self.expert_biases = None
                self.bias = None
        elif self.emb_type == 'SharedMoELoRAEmb':
            self._init_moe_router(n, input_dim_for_dynamic_mlp=input_dim)
            assert self.rank is not None

            self.shared_weight = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.normal_(self.shared_weight, mean=0.0, std=1e-2)
            self.lora_A_experts = nn.Parameter(torch.empty(self.num_experts, input_dim, self.rank))
            self.lora_B_experts = nn.Parameter(torch.empty(1, self.rank, output_dim))
            nn.init.normal_(self.lora_A_experts, mean=0.0, std=1e-5)
            nn.init.zeros_(self.lora_B_experts)

            if bias:
                if self.grouped_bias:
                    self.expert_biases = nn.Parameter(torch.empty(self.num_experts, output_dim))
                    nn.init.normal_(self.expert_biases,mean=0.0,std=1e-2)
                    self.bias = None
                else:
                    self.expert_biases = None
                    self.bias = nn.Parameter(torch.empty(n,output_dim))
                    nn.init.zeros_(self.bias)
            else:
                self.expert_biases = None
                self.bias = None
            
        elif self.emb_type == 'MoELoRAEmb':
            self._init_moe_router(n, input_dim_for_dynamic_mlp=input_dim)
            assert self.rank is not None

            self.lora_A_experts = nn.Parameter(torch.empty(self.num_experts, input_dim, self.rank))
            self.lora_B_experts = nn.Parameter(torch.empty(1, self.rank, output_dim))
            nn.init.normal_(self.lora_A_experts, mean=0.0, std=1e-2)
            nn.init.zeros_(self.lora_B_experts)

            if bias:
                if self.grouped_bias:
                    self.expert_biases = nn.Parameter(torch.empty(self.num_experts, output_dim))
                    nn.init.normal_(self.expert_biases,mean=0.0,std=1e-2)
                    self.bias = None
                else:
                    self.expert_biases = None
                    self.bias = nn.Parameter(torch.empty(n, output_dim))
                    nn.init.zeros_(self.bias)
            else:
                self.expert_biases = None
                self.bias = None
            


    def _init_moe_router(self, n, input_dim_for_dynamic_mlp=None):
        assert self.num_experts is not None, f"num_experts must be specified for {self.emb_type}"
        assert self.moe_router_type in ['learned', 'mlp_id', 'mlp_seq'], \
            f"moe_router_type for {self.emb_type} must be 'learned', 'mlp_id', or 'mlp_seq'"

        if self.moe_router_type == 'learned':
            self.gate_weights = nn.Parameter(torch.empty(n, self.num_experts))
            if self.use_softmax:
                nn.init.normal_(self.gate_weights, mean=0.0, std=1e-2)
            else:
                nn.init.normal_(self.gate_weights, mean=1.0 / self.num_experts, std=1.0/self.num_experts/10)
        elif self.moe_router_type == 'mlp_id':
            assert self.channel_identity_dim is not None, \
                f"channel_identity_dim must be provided in constructor for {self.emb_type} when moe_router_type is 'mlp_id'"
            self.channel_identities = nn.Parameter(torch.empty(n, self.channel_identity_dim))
            nn.init.normal_(self.channel_identities, mean=0.0, std=1e-2)

            assert self.moe_mlp_hidden_dim is not None, \
                f"moe_mlp_hidden_dim must be provided for {self.emb_type} with moe_router_type 'mlp_id'"
            
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.channel_identity_dim, self.moe_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(self.moe_mlp_hidden_dim, self.num_experts)
            )
            final_layer = self.gate_mlp[-1]
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
            if self.use_softmax:
                nn.init.zeros_(final_layer.bias)
            else:
                nn.init.constant_(final_layer.bias, 1.0 / self.num_experts)
        elif self.moe_router_type == 'mlp_seq':
            assert input_dim_for_dynamic_mlp is not None, \
                f"input_dim_for_dynamic_mlp must be passed to _init_moe_router for {self.emb_type} with moe_router_type 'mlp_seq'"
            assert self.moe_mlp_hidden_dim is not None, \
                f"moe_mlp_hidden_dim must be provided for {self.emb_type} with moe_router_type 'mlp_seq'"
            self.gate_mlp = nn.Sequential(
                nn.Linear(input_dim_for_dynamic_mlp, self.moe_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(self.moe_mlp_hidden_dim, self.num_experts)
            )
            final_layer = self.gate_mlp[-1]
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
            if self.use_softmax:
                nn.init.zeros_(final_layer.bias)
            else:
                nn.init.constant_(final_layer.bias, 1.0 / self.num_experts)

    def _apply_gating(self, gating_scores, dim=-1):
        if self.use_softmax:
            return F.softmax(gating_scores, dim=dim)
        else:
            return gating_scores

    def forward(self, x):
        B, C, d_i = x.shape
        x = x.transpose(0, 1)

        if self.emb_type == 'SharedEmb':
            x = x @ self.weight
        elif self.emb_type == 'IndEmb':
            x = x @ self.weight
        elif self.emb_type == 'SharedIndEmb':
            x = x @ (self.shared_weight + self.ind_weight)
        elif self.emb_type == 'SharedIndLoRAEmb':
            shared_out = x @ self.shared_weight
            lora_out = (x @ self.lora_A) @ self.lora_B
            x = shared_out + lora_out
        elif self.emb_type == 'IndLoRAEmb':
            x = (x @ self.lora_A) @ self.lora_B
        elif self.emb_type == 'MoEEmb':
            if self.moe_router_type == 'learned' or self.moe_router_type == 'mlp_id':
                use_scheme_b = d_i < B * self.num_experts if self.force_scheme is None else self.force_scheme == 'B'

                if self.moe_router_type == 'learned':
                    gating_scores = self.gate_weights
                else:
                    gating_scores = self.gate_mlp(self.channel_identities)
                gating_probs = self._apply_gating(gating_scores)

                if use_scheme_b:
                    effective_expert_weight = torch.einsum('cn,nio->cio', gating_probs, self.experts)
                    x_out = torch.einsum('cbi,cio->cbo', x, effective_expert_weight)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        x_out = x_out + gated_bias
                    x = x_out
                else:
                    expert_outputs = torch.einsum('cbi,nio->cbno', x, self.experts)
                    x_out = torch.einsum('cn,cbno->cbo', gating_probs, expert_outputs)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        x_out = x_out + gated_bias
                    x = x_out
            elif self.moe_router_type == 'mlp_seq':
                gating_scores = self.gate_mlp(x)
                gating_probs = self._apply_gating(gating_scores)
                expert_outputs = torch.einsum('cbi,nio->cbno', x, self.experts)
                x_out = torch.einsum('cbn,cbno->cbo', gating_probs, expert_outputs)
                if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                    gated_bias = torch.einsum('cbn,no->cbo', gating_probs, self.expert_biases)
                    x_out = x_out + gated_bias
                x = x_out
            else:
                   raise ValueError(f"Unknown moe_router_type: {self.moe_router_type}")

        elif self.emb_type == 'SharedMoEEmb':
            if self.moe_router_type == 'learned' or self.moe_router_type == 'mlp_id':
                use_scheme_b = d_i < B * self.num_experts if self.force_scheme is None else self.force_scheme == 'B'
                
                if self.moe_router_type == 'learned':
                    gating_scores = self.gate_weights
                else:
                    gating_scores = self.gate_mlp(self.channel_identities)
                gating_probs = self._apply_gating(gating_scores)

                if use_scheme_b:
                    effective_moe_weight = torch.einsum('cn,nio->cio', gating_probs, self.experts)
                    total_weight = self.shared_weight.unsqueeze(0) + effective_moe_weight
                    x_out = torch.einsum('cbi,cio->cbo', x, total_weight)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        x_out = x_out + gated_bias
                    x = x_out
                else:
                    shared_out = x @ self.shared_weight
                    expert_outputs = torch.einsum('cbi,nio->cbno', x, self.experts)
                    moe_out = torch.einsum('cn,cbno->cbo', gating_probs, expert_outputs)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        moe_out = moe_out + gated_bias
                    x = shared_out + moe_out
            elif self.moe_router_type == 'mlp_seq':
                shared_out = x @ self.shared_weight
                gating_scores = self.gate_mlp(x)
                gating_probs = self._apply_gating(gating_scores)
                expert_outputs = torch.einsum('cbi,nio->cbno', x, self.experts)
                moe_out = torch.einsum('cbn,cbno->cbo', gating_probs, expert_outputs)
                if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                    gated_bias = torch.einsum('cbn,no->cbo', gating_probs, self.expert_biases)
                    moe_out = moe_out + gated_bias
                x = shared_out + moe_out
            else:
                raise ValueError(f"Unknown moe_router_type: {self.moe_router_type}")

        elif self.emb_type == 'SharedMoELoRAEmb':
            if self.moe_router_type == 'learned' or self.moe_router_type == 'mlp_id':
                use_scheme_b = d_i < B * self.num_experts if self.force_scheme is None else self.force_scheme == 'B'

                if self.moe_router_type == 'learned':
                    gating_scores = self.gate_weights
                else:
                    gating_scores = self.gate_mlp(self.channel_identities)
                gating_probs = self._apply_gating(gating_scores)

                if use_scheme_b:
                    A_eff = torch.einsum('ce,eir->cir', gating_probs, self.lora_A_experts)
                    W_lora = torch.einsum('cir,ro->cio', A_eff, self.lora_B_experts[0])
                    W_total = self.shared_weight.unsqueeze(0) + W_lora
                    x_out = torch.einsum('cbi,cio->cbo', x, W_total)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        x_out = x_out + gated_bias
                    x = x_out
                else:
                    shared_out = x @ self.shared_weight
                    lora_A_out = torch.einsum('cbi,nir->cbnr', x, self.lora_A_experts)
                    lora_expert_outputs = torch.einsum('cbnr,nro->cbno', lora_A_out, self.lora_B_experts)
                    moe_lora_out = torch.einsum('cn,cbno->cbo', gating_probs, lora_expert_outputs)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        moe_lora_out = moe_lora_out + gated_bias
                    x = shared_out + moe_lora_out
            elif self.moe_router_type == 'mlp_seq':
                shared_out = x @ self.shared_weight
                gating_scores = self.gate_mlp(x)
                gating_probs = self._apply_gating(gating_scores)
                lora_A_out = torch.einsum('cbi,nir->cbnr', x, self.lora_A_experts)
                lora_expert_outputs = torch.einsum('cbnr,nro->cbno', lora_A_out, self.lora_B_experts)
                moe_lora_out = torch.einsum('cbn,cbno->cbo', gating_probs, lora_expert_outputs)
                if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                    gated_bias = torch.einsum('cbn,no->cbo', gating_probs, self.expert_biases)
                    moe_lora_out = moe_lora_out + gated_bias
                x = shared_out + moe_lora_out
            else:
                raise ValueError(f"Unknown moe_router_type: {self.moe_router_type}")

        elif self.emb_type == 'MoELoRAEmb':
            if self.moe_router_type == 'learned' or self.moe_router_type == 'mlp_id':
                use_scheme_b = d_i < B * self.num_experts if self.force_scheme is None else self.force_scheme == 'B'

                if self.moe_router_type == 'learned':
                    gating_scores = self.gate_weights
                else:
                    gating_scores = self.gate_mlp(self.channel_identities)
                gating_probs = self._apply_gating(gating_scores)

                if use_scheme_b:
                    A_eff = torch.einsum('ce,eir->cir', gating_probs, self.lora_A_experts)
                    W_lora_eff = torch.einsum('cir,ro->cio', A_eff, self.lora_B_experts[0])
                    x_out = torch.einsum('cbi,cio->cbo', x, W_lora_eff)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        x_out = x_out + gated_bias
                    x = x_out
                else:
                    lora_A_out = torch.einsum('cbi,nir->cbnr', x, self.lora_A_experts)
                    lora_expert_outputs = torch.einsum('cbnr,nro->cbno', lora_A_out, self.lora_B_experts)
                    moe_lora_out = torch.einsum('cn,cbno->cbo', gating_probs, lora_expert_outputs)
                    if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                        gated_bias = torch.einsum('cn,no->co', gating_probs, self.expert_biases).unsqueeze(1)
                        moe_lora_out = moe_lora_out + gated_bias
                    x = moe_lora_out
            elif self.moe_router_type == 'mlp_seq':
                gating_scores = self.gate_mlp(x)
                gating_probs = self._apply_gating(gating_scores)
                lora_A_out = torch.einsum('cbi,nir->cbnr', x, self.lora_A_experts)
                lora_expert_outputs = torch.einsum('cbnr,nro->cbno', lora_A_out, self.lora_B_experts)
                moe_lora_out = torch.einsum('cbn,cbno->cbo', gating_probs, lora_expert_outputs)
                if hasattr(self, 'expert_biases') and self.expert_biases is not None:
                    gated_bias = torch.einsum('cbn,no->cbo', gating_probs, self.expert_biases)
                    moe_lora_out = moe_lora_out + gated_bias
                x = moe_lora_out
            else:
                raise ValueError(f"Unknown moe_router_type: {self.moe_router_type}")

        else:
            raise NotImplementedError

        x = x.transpose(0, 1)

        if self.bias is not None:
            x = x + self.bias

        return x