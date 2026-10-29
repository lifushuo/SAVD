import torch
import torch.nn as nn
import torchvision.models as models

class TSP_ResNet_Policy_Rigorous(nn.Module):
    def __init__(self, grid_size=5): 
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) 
        
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False
                
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((grid_size, grid_size)), 
            
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.policy_head(x)
        return x.view(x.size(0), -1) 
# import torch
# import torch.nn as nn
# import torchvision.models as models

# class TSP_ResNet_Policy_Rigorous(nn.Module):
#     def __init__(self, grid_size=5): 
#         super().__init__()
#         self.backbone = models.resnet18(pretrained=True) 
#         for param in list(self.backbone.children())[:4]:
#             param.requires_grad = False
            
#         self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
#         self.policy_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((grid_size, grid_size)), 
            
#             nn.Conv2d(512, 128, kernel_size=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(128, 1, kernel_size=1) 
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.policy_head(x)
#         return x.view(x.size(0), -1) # -> [B, 25]
# import torch
# import torch.nn as nn
# import torchvision.models as models

# class TSP_ResNet_Policy(nn.Module):
#     def __init__(self, grid_size=5): 
#         super().__init__()
        
#         self.backbone = models.resnet18(pretrained=False) 
        
#         self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
#         self.policy_head = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(128, 1, kernel_size=1) 
#         )

#     def forward(self, x):
#         # x: [B, 3, 640, 640]
#         x = self.features(x)      # -> [B, 512, 20, 20]
#         x = self.policy_head(x)   # -> [B, 1, 5, 5]
        
#         return x.view(x.size(0), -1)
    
# import torch
# import torch.nn as nn
# import torchvision.models as models

class TSP_ResNet_Policy(nn.Module):
    def __init__(self, grid_size=5): 
        super().__init__()
        
        self.backbone = models.resnet18(pretrained=False) 
        
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)), # [B, 512, 5, 5]
            
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 1, 1) # [B, 1, 5, 5]
        )

    def forward(self, x):
        # x: [B, 3, 640, 640]
        x = self.features(x)      
        x = self.policy_head(x)   # -> [B, 1, 5, 5]
        
        return x.view(x.size(0), -1)
    
import torch
import torch.nn as nn
import torchvision.models as models


class HybridPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):

        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)

        return torch.cat([avg_feat, max_feat], dim=1)


# class TSP_ResNet_Policy_27(nn.Module):

#     def __init__(self, grid_size=5):
#         super().__init__()

#         # =========================
#         # Backbone
#         # =========================
#         self.backbone = models.resnet18(pretrained=True)

#         for name, param in self.backbone.named_parameters():
#             if "layer3" not in name and "layer4" not in name:
#                 param.requires_grad = False

#         self.features = nn.Sequential(
#             *list(self.backbone.children())[:-2]
#         )

#         # =========================
#         # Policy Head
#         # =========================
#         self.policy_head = nn.Sequential(

#             # ---------------------------------
#             # ---------------------------------
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             # ---------------------------------
#             # 2. Hybrid Pooling
#             # ---------------------------------
#             HybridPooling((grid_size, grid_size)),

#             # ---------------------------------
#             # ---------------------------------
#             nn.Conv2d(512, 128, kernel_size=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),

#             # ---------------------------------
#             # ---------------------------------
#             nn.Conv2d(128, 1, kernel_size=1)
#         )

#     def forward(self, x):

#         x = self.features(x)

#         x = self.policy_head(x)

#         return x.view(x.size(0), -1)
import torch
import torch.nn as nn
import torchvision.models as models

class HybridDownsample(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        self.learned_downsample = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f_avg = self.avg_pool(x)                # [B, 512, 5, 5]
        f_learned = self.learned_downsample(x)  # [B, 128, 5, 5]
        
        return torch.cat([f_avg, f_learned], dim=1) # [B, 640, 5, 5]


class TSP_ResNet_Policy_27(nn.Module):
    def __init__(self, grid_size=5):
        super().__init__()
        
        # =========================
        # =========================
        self.backbone = models.resnet18(pretrained=True)

     
        self.features = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        # =========================
        # =========================
        self.hybrid_pool = HybridDownsample(in_channels=512)

        # =========================
        # =========================
        self.policy_head = nn.Sequential(
            nn.Conv2d(512 + 128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)         # -> [B, 512, 35, 35]
        x = self.hybrid_pool(x)      # -> [B, 640, 5, 5]
        x = self.policy_head(x)      # -> [B, 1, 5, 5]
        
        return x.view(x.size(0), -1) # -> [B, 25]