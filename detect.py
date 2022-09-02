import os
import torch
import argparse

from ModelBYCC import ConfigBYCCComplete, SimpleNetworkBYCC


def detect():
    files = os.listdir("train")
    files = ["train/" + item for item in files if ".pt" in item]
    print(files)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--mode", type=str, default="origin", help="continue or origin")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=100, help="save_step")
    parser.add_argument("--seed", type=int, default=100, help="seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = []
    for i, one_path in enumerate(files):
        try:
            config = ConfigBYCCComplete()
            model = SimpleNetworkBYCC(config, args).to(device)

            model.load_state_dict(torch.load(one_path, map_location=device)["model_state_dict"])
            model.eval()
            print("{} path: {}".format(i, one_path))
            t = model.x
            y = model(t)

            Cln = y[:, 0]
            ClbSt = y[:, 1]
            MBF = y[:, 2]
            Nrm1t = y[:, 3]
            ClbMt = y[:, 4]
            Polo = y[:, 5]
            Sic1t = y[:, 6]
            SBF = y[:, 7]
            Cdh1 = y[:, 8]
            Cdc14 = y[:, 9]
            loss, _ = model.loss()
            real_loss = model.real_loss()
            sets.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i, loss.item(), real_loss.item(),
                                       Cln[-1].cpu().detach().numpy(),
                                       ClbSt[-1].cpu().detach().numpy(),
                                       MBF[-1].cpu().detach().numpy(),
                                       Nrm1t[-1].cpu().detach().numpy(),
                                       ClbMt[-1].cpu().detach().numpy(),
                                       Polo[-1].cpu().detach().numpy(),
                                       Sic1t[-1].cpu().detach().numpy(),
                                       SBF[-1].cpu().detach().numpy(),
                                       Cdh1[-1].cpu().detach().numpy(),
                                       Cdc14[-1].cpu().detach().numpy(),
                                       ))
        except Exception as e:
            print(e)
    for item in sets:
        print(item)

def Eq(x, y):
    return x

from math import sqrt

def validate():
    Cln, ClbSt, MBF, Nrm1t, ClbMt, Polo, Sic1t, SBF, Cdh1, Cdc14 = 0.046850682, 0.011319331, 0.012099141, 0.489256892, 0.65124172, 0.24496382, 0.015620268, 0.047214321, 0.021127737, 0.344143702
    kscln = 0.2
    kdcln = 0.2
    kasbf1 = 1
    kasbf = 10
    kisbf = 25
    Jsbf = 1
    ksclbs = 0.15
    kdclbs1 = 0.1
    kdclbs = 0.05
    ksnrm1 = 0.05
    kdnrm1 = 0.1
    MBFtot = 0.5
    kass1 = 1
    kdiss1 = 0.001
    Jmbf = 0.01
    ksclbm1 = 0.01
    ksclbm = 0.01
    kdclbm1 = 0.01
    kdclbm = 1
    Jclbm = 0.05
    n = 2
    kspolo = 0.01
    kdpolo1 = 0.01
    kdpolo = 1
    kacdc14 = 1
    kicdc14 = 0.25
    Jcdc14 = 0.01
    kssic1 = 0.02
    kdsic1 = 0.01
    kdsic = 2
    Jsic1 = 0.01
    Kdiss = 0.05
    kacdh11 = 1
    kacdh1 = 10
    kicdh11 = 0.2
    kicdh1 = 10
    Jcdh1 = 0.01
    ndClbM = 0

    vals = []

    eq1 = Eq((kscln * SBF - kdcln * Cln), 0.0)
    eq2 = Eq((ksclbs * (MBF * Cln / (Jmbf + Cln)) - (kdclbs1 + kdclbs * Cdh1) * ClbSt), 0.0)
    eq3 = Eq((kdiss1 * (MBFtot - MBF) - kass1 * MBF * (Nrm1t - (MBFtot - MBF))), 0.0)
    eq4 = Eq((ksnrm1 * (MBF * Cln / (Jmbf + Cln)) - kdnrm1 * Cdh1 * Nrm1t), 0.0)
    eq5 = Eq((ksclbm1 + ksclbm * ((ClbMt + ndClbM) * ((ClbSt + ClbMt + ndClbM) - (
                2 * Sic1t * (ClbSt + ClbMt + ndClbM) / ((Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
            (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM))))) / (
                                              ClbSt + ClbMt + ndClbM)) ** n / (Jclbm ** n + ((ClbMt + ndClbM) * (
                (ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                    (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM))))) / (
                                                                                                         ClbSt + ClbMt + ndClbM)) ** n) - (
                          kdclbm1 + kdclbm * Cdh1) * ClbMt), 0.0)
    eq6 = Eq((kspolo * ((ClbMt + ndClbM) * ((ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
            (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM))))) / (
                                    ClbSt + ClbMt + ndClbM)) - (kdpolo1 + kdpolo * Cdh1) * Polo), 0.0)
    eq7 = Eq((kssic1 - (kdsic1 + kdsic * ((ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
            (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM))))) * (Cln + (
                (ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                    (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM)))))) / (
                                    Jsic1 + Cln + ((ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                                        (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
                                    (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (
                                                ClbSt + ClbMt + ndClbM))))))) * Sic1t), 0.0)
    eq8 = Eq(((kasbf1 + kasbf * Cln) * (1 - SBF) / (Jsbf + 1 - SBF) - kisbf * ((ClbMt + ndClbM) * (
                (ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                    (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM))))) / (
                                                                                           ClbSt + ClbMt + ndClbM)) * SBF / (
                          Jsbf + SBF)), 0.0)
    eq9 = Eq(((kacdh11 + kacdh1 * Cdc14) * (1 - Cdh1) / (Jcdh1 + 1 - Cdh1) - (kicdh11 * Cln + kicdh1 * (
                (ClbSt + ClbMt + ndClbM) - (2 * Sic1t * (ClbSt + ClbMt + ndClbM) / (
                    (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) + sqrt(
                (Sic1t + (ClbSt + ClbMt + ndClbM) + Kdiss) ** 2 - 4 * Sic1t * (ClbSt + ClbMt + ndClbM)))))) * Cdh1 / (
                          Jcdh1 + Cdh1)), 0.0)
    eq10 = Eq((kacdc14 * Polo * (1 - Cdc14) / (Jcdc14 + 1 - Cdc14) - kicdc14 * Cdc14 / (Jcdc14 + Cdc14)), 0.0)
    vals = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10]
    vals = [str(item) for item in vals]
    print("\t".join(vals))


if __name__ == "__main__":
    # detect()
    validate()
