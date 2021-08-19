
def train_model(model, criterion_binary, criterion_regression, optimizer, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs):
        train_loss = 0.0
        train_loss_age = 0.0
        train_loss_gender = 0.0
        valid_loss = 0.0

        # train the model 
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label_age, label_gender = sample_batched['image'].to(device),\
                                             sample_batched['label_age'].to(device),\
                                              sample_batched['label_gender'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(image)
            label_age_hat = output['age']
            label2_gender_hat = output['gender']
     
            # calculate loss
            loss_age = criterion_regression(label_age_hat, label_age)
            loss_gender = criterion_binary(label2_gender_hat, label_gender.unsqueeze(-1))

      
            loss = loss_age + loss_gender
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss_age = train_loss_age + ((1 / (batch_idx + 1)) * (loss_age.data - train_loss_age))
            train_loss_gender = train_loss_gender + ((1 / (batch_idx + 1)) * (loss_gender.data - train_loss_gender))

            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d, loss: %.6f loss_age: %.6f, loss_gender: %.6f' %
                  (epoch, batch_idx + 1, train_loss, loss_age, loss_gender))
                
        # validate the model #
        model.eval()
        for batch_idx, sample_batched in enumerate(test_dataloader):
            image, label_age, label_gender = sample_batched['image'].to(device),\
                                             sample_batched['label_age'].to(device),\
                                              sample_batched['label_gender'].to(device)

            output = model(image)
            label_age_hat = output['age']
            label2_gender_hat = output['gender']
         
            # calculate loss
            loss_age = criterion_regression(label_age_hat, label_age)
            loss_gender = criterion_binary(label2_gender_hat, label_gender.unsqueeze(-1))


            loss = loss_age + loss_gender
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss
                ))

            valid_loss_min = valid_loss
            
    # return trained model
    return model