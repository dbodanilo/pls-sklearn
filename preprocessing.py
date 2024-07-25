from model import IV, LS, WS, PAIRS, load_deap


IDS = tuple("(I_D)" + p for p in PAIRS)
WLS = tuple("(W/L)" + p for p in PAIRS)
IDWLS = tuple("(I_{D}/(W/L))" + p for p in PAIRS)


def scale_test(X, mean, std):
    X_scaled = X - mean
    X_scaled /= std

    return X_scaled


def scale_transform(scaler, model, X):
    return model.transform(scaler.transform(X))


def extend_deap(loader=load_deap):
    deap_data, _ = loader(idwl=False, split=False)

    for i, wl in enumerate(WLS):
        deap_data[wl] = deap_data[WS[i]] / deap_data[LS[i]]

    # (I_D)_{9} = I_{pol}
    deap_data[IDS[4]] = deap_data[IV[0]]

    # (I_D)_{10} = (I_D)_{9} * (W/L)_{10} / (W/L)_{9}
    deap_data[IDS[5]] = deap_data[IDS[4]] * \
        deap_data[WLS[5]] / deap_data[WLS[4]]

    # (I_D)_{1,2} = (I_D)_{10} / 2
    deap_data[IDS[0]] = deap_data[IDS[5]] / 2

    # (I_D)_{3,4} = (I_D)_{1,2}
    deap_data[IDS[1]] = deap_data[IDS[0]]

    # (I_D)_{5,6} = (I_D)_{3,4} * (W/L)_{5,6} / (W/L)_{3,4}
    deap_data[IDS[2]] = deap_data[IDS[1]] * \
        deap_data[WLS[2]] / deap_data[WLS[1]]

    # (I_D)_{7,8} = (I_D)_{5,6}
    deap_data[IDS[3]] = deap_data[IDS[2]]

    for i, idwl in enumerate(IDWLS):
        deap_data[idwl] = deap_data[IDS[i]] / deap_data[WLS[i]]

    deap_data.drop([*IDS, *WLS], axis="columns", inplace=True)

    return deap_data
