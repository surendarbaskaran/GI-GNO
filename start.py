import training
import preprocessing
import inference
import warnings
import config
import model
import time

warnings.filterwarnings("ignore") 



def print_config(cfg):
    print("\n" + "="*50)
    print("🚀 APPLICATION CONFIGURATION")
    print("="*50)

    params = {
        k: v for k, v in cfg.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }

    for k, v in params.items():
        print(f"{k:<20} : {v}")

    print("="*50 + "\n")


def main():
    print_config(config)

    print("Starting training pipeline...\n")

    pipeline_start=time.time()

    preprocessing.main()
    training.main()
    inference.main()

    time_=time.time()-pipeline_start

    print(f"script completed , time taken : {time:.2f} sec")


if __name__ == "__main__":
    main()

