import torch

if __name__ == "__main__":
    print(f"cuda is avalable{torch.cuda.is_available()}")
    print(f"cuda device count is {torch.cuda.device_count()}")
    # print(f"cuda device name is {torch.cuda.current_device()}")
    print(f"torch version is {torch.__version__}")
