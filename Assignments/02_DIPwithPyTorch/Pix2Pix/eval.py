import torch
from FCN_network import FullyConvNetwork
import argparse
from facades_dataset import FacadesDataset
from tqdm import tqdm
from torchvision.utils import save_image


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the pixel values are in the range [0, 1]
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value.item()

def ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)

    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))

    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_value.item()

def generate_test_listfile(test_dir):
    import os
    test_list_path = os.path.join(test_dir, 'test_list.txt')
    with open(test_list_path, 'w') as f:
        for filename in os.listdir(test_dir):
            if filename.endswith(('.jpg', '.png')):
                f.write(os.path.join(test_dir, filename) + '\n')
    return test_list_path

def main():
    # args
    parser = argparse.ArgumentParser(description='Evaluate the model on the test dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, default='datasets/facades/test', help='Path to the test list file')
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save test results")
    
    args = parser.parse_args()
    # generate file list
    test_list_path = generate_test_listfile(args.test_dir)
    print(f"Test list file generated at: {test_list_path}")

    # load the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FullyConvNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # load the dataset
    dataset = FacadesDataset(list_file=test_list_path, augment=False)
    
    # evaluation loop
    psnr_values = []
    ssim_values = []
    mean_err_values = []
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            input_image, target_image = dataset[i]
            input_image = input_image.unsqueeze(0).to(device)
            target_image = target_image.unsqueeze(0).to(device)
            output_image = model(input_image)[0]
            psnr_values.append(psnr(output_image, target_image))
            ssim_values.append(ssim(output_image, target_image))
            mean_err_values.append((output_image - target_image).abs().mean().item())
            # horizontally concat input, output, target and save
            concat_image = torch.cat((input_image.cpu(), output_image.cpu(), target_image.cpu()), dim=3)
            save_image(concat_image /2. + 0.5, os.path.join(args.output_dir, 'images', f'test_{i}.png'))
            
    print(f"Average PSNR: {sum(psnr_values) / len(psnr_values)}")
    print(f"Average SSIM: {sum(ssim_values) / len(ssim_values)}")
    
    result_json = {
        "PSNR": sum(psnr_values) / len(psnr_values),
        "SSIM": sum(ssim_values) / len(ssim_values),
        "MEAN": sum(mean_err_values) / len(mean_err_values)
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        import json
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    main()