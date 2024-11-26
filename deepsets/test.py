import torch

tensor_feature = torch.tensor([[[897.0629, 715.2148],
                             [950.2504, 736.0510]],
                            [[1029.1488, 709.4039],
                             [736.8854, 668.9133]],
                            [[1007.2249, 705.7989],
                             [713.9932, 734.6144]]])

pooled  = torch.tensor([[[127.6998,   0.0000, 426.9844,   0.0000,   0.0000, 101.9602, 464.6452,
            0.0000]],

        [[122.3882,   0.0000, 407.0455,   0.0000,   0.0000,  96.9140, 443.2073,
            0.0000]],

        [[116.0228,   0.0000, 408.9862,   0.0000,   0.0000, 100.5146, 442.5656,
            0.0000]]])

pooled_expanded = pooled.expand(-1, tensor_feature.size(1), -1)


result = torch.cat([tensor_feature, pooled_expanded], dim=2)
print(result)






tensor_3d = tensor_3d.expand(2, -1)

print(tensor_3d)