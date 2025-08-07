def fit_model(data: dict, n_epochs: int = 1000, learning_rate: float = 1e-2, device: str="cpu", tol: float=1e-4) -> Tuple[dict]:
    """
    params:
        data: dict
            dictionary containing the data
        n_epochs: int
            number of epochs to train the model
        learning_rate: float
            learning rate for the optimizer
    """

    # initialize the model
    model = OptimizationModel(data)
    model = model.to(device)

    # initialize the optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)



    # define the loss function
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    epochs_since_best = 0
    best_dwell_times = None
    loss_list = []
    tolerance = tol
    tolerance_count = 0

    tic = time()
    dydx_list = []
   
    # training loop
    for epoch in tqdm(range(n_epochs)):
        # zero the gradients
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
        # forward pass
            dwell_time_scalars = model(data["dwell_positions"])
            predicted_dose = torch.sum(data["dwell_positions"] * dwell_time_scalars[:,None,None,None], dim=0)
            loss = loss_fn(predicted_dose[data["true_dose_mask"]], data["true_dose"])

            if(epoch % 100 == 0):
                print(dwell_time_scalars)
                print(loss.item())
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()

        # total_tolerance = tol + tol*model.args_dict["weights"]["hrctv"]["lower"] + tol*model.args_dict["weights"]["hrctv"]["upper"] + tol*model.args_dict["weights"]["bladder"] + tol*model.args_dict["weights"]["rectum"] + tol*model.args_dict["weights"]["sigmoid"]
        # print the loss
        # if epoch % 1000 == 0:

        #     print(f"Epoch {epoch}, Loss {loss.item()}")
        #     # print(f"Current dwell time scalars: {np.round(model.dwell_time_scalars.detach().to("cpu").numpy(),2)}")
        #     print(f"Current dwell time scalars: {model.dwell_time_scalars.detach().to("cpu").numpy()}")

    # Add original ring data back in for calculation

    dwell_time_scalars = model.dwell_time_scalars.detach()
