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

    start=time.time()

    # preprocessing.main()
    training.main()
    # inference.main()
    result=inference.run_inference(
        "dataset/testvtk/case_0473.vtk",
        {"geom_name": "geom_047",
        "alt_kft": 2.541,
        "Re": 21630000,
        "M_inf": 0.4881,
        "alpha_deg": 11.05,
        "beta_deg": 0,}
    )
    print(result)

    elapsed = time.time() - start

    print(f"script completed , time taken : {elapsed:.2f} sec")


if __name__ == "__main__":
    main()

