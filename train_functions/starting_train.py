import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    print(batch_size)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent

            # might need some changes
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # might need some changes
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar('train_loss', loss, global_step=step)
                step += 1

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                print(evaluate(val_loader, model, loss_fn))

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    #argmax (maybe)
    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    
    TODO!
    """
    sum_acc = 0
    counter = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            outputs = torch.argmax(outputs, dim = 1)
            sum_acc += compute_accuracy(outputs, labels)
            counter += 1
        
        sum_acc /= counter
        sum_acc *= 100
            
    return sum_acc



"""
for epoch in range(5):
    for batch in train_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = conv_net(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""