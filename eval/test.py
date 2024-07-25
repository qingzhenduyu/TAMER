import os
import json
import typer
from pytorch_lightning import Trainer, seed_everything

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

seed_everything(7)
years = {'2014': 986, '2016': 1147, '2019': 1199, 'test': 24607}


def main(
    folder: str, version: str, test_year: str, max_size: int, scale_to_limit: bool
):
    ckp_folder = os.path.join(
        "lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = HMEDatamodule(
        folder=folder,
        test_folder=test_year,
        max_size=max_size,
        scale_to_limit=scale_to_limit,
    )

    model = LitTAMER.load_from_checkpoint(ckp_path)

    metrics = trainer.test(model, datamodule=dm)[0]

    with open(os.path.join(ckp_folder, os.pardir, f"{test_year}.txt"), "w") as f:
        for tol, acc in metrics.items():
            f.write(f"Exprate {tol} tolerated: {acc:.3f}\n")

    os.rename(
        "errors.json",
        os.path.join(ckp_folder, os.pardir, f"errors_{test_year}.json"),
    )
    os.rename(
        "predictions.json",
        os.path.join(ckp_folder, os.pardir, f"pred_{test_year}.json"),
    )

    # Calculate ExpRate
    test_num = years[test_year]
    with open(os.path.join(ckp_folder, os.pardir, f"errors_{test_year}.json"), 'r') as jf:
        data = json.load(jf)
        exprate = test_num-len(data)
        exprate_1 = 0
        exprate_2 = 0
        for _, ele in data.items():
            if (ele['dist']) <= 1:
                exprate_1 += 1
            if (ele['dist']) <= 2:
                exprate_2 += 1
        exprate_1 = (exprate_1 + exprate)/test_num
        exprate_2 = (exprate_2 + exprate)/test_num
        exprate = exprate/test_num
        with open(os.path.join(ckp_folder, os.pardir, f'{test_year}.txt'), 'w') as wf:
            wf.write(f'ExpRate:  {exprate}\n')
            wf.write(f'ExpRate<=1:  {exprate_1}\n')
            wf.write(f'ExpRate<=2:  {exprate_2}\n')


if __name__ == "__main__":
    typer.run(main)
