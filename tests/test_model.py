import torch

from handaffordsplat.model import PointNetMean


def test_pointnet_mean_output_shape():
    model = PointNetMean(input_dim=401, latent_dim=192)
    output = model(torch.randn(11, 401))
    assert output.shape == (11,)


def test_pointnet_mean_uses_object_context():
    torch.manual_seed(4)
    model = PointNetMean(input_dim=401, latent_dim=192)
    model.eval()
    point = torch.randn(1, 401)
    with torch.no_grad():
        alone = model(point)[0]
        with_context = model(torch.cat([point, torch.randn(5, 401)], dim=0))[0]
    assert not torch.isclose(alone, with_context)
