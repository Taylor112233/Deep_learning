% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

% number of training samples ;dimension of feature vector
   [N D]= size(train_set);
% number of input, hidden and output nodes
    input_nodes = D;
    hidden_nodes = 300;
    output_nodes = 10;

% learning rate
    learning_rate = 0.05;
% scaling factor for sigmoid function
    beta = 0.01; 

% create instance of neural network
    % the weights between input and hidden layer 
        wih = randn(1+input_nodes,hidden_nodes)*sqrt(1/hidden_nodes);
    % the weights between hidden and output layer
        who = randn(1+hidden_nodes,output_nodes)*sqrt(1/output_nodes);

    
% train the neural network

    %epochs is the number of times the training data set is used for training
    epochs = 2;

for i=1:epochs
    
    disp([num2str(i), ' epochs']);
    
    for j=1:N
        
        % propagate the input forward through the network
        inputs = [1; train_set(j, :)'];
        
        targets = eye(output_nodes);

        % calculate signals into hidden layer
        hidden_inputs = wih' * inputs;
        % calculate the signals emerging from hidden layer
        hidden_outputs = [1;sigmf(hidden_inputs, [beta 0])];
        
        % calculate signals into final output layer
        final_inputs = who' * hidden_outputs;
        % calculate the signals emerging from final output layer
        final_outputs = sigmf(final_inputs, [beta 0]);
        
        % propagate the error backward through the network
        % output layer error is the (target - actual)
        output_errors = (final_outputs - targets(:,train_label(j)+1)).* final_outputs .* (1.0 - final_outputs);
        % hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = (who* output_errors) .* hidden_outputs .* (1.0 - hidden_outputs);
        hidden_errors = hidden_errors(2:end);

        % update the weights for the links between the hidden and output layers
        who = who - learning_rate * (hidden_outputs * output_errors');
        % update the weights for the links between the input and hidden layers
        wih = wih - learning_rate * (inputs * hidden_errors');
       
    end

end


% test the neural network
    
% go through all the records in the test data set
    test_size = size(test_set);
    num_correct = 0;
    
    for i=1:test_size(1)
       input_x = [1; test_set(i,:)'];
       
        % query the neural network
        % calculate signals into hidden layer
        hidden_input = wih' * input_x;
        % calculate the signals emerging from hidden layer
        hidden_output = [1; sigmf(hidden_input, [beta 0])];

        % calculate signals into final output layer
        final_input = who' * hidden_output;
        % calculate the signals emerging from final output layer
        final_output = sigmf(final_input, [beta 0]);
        
        [max_unit, max_idx] = max(final_output);
        if(max_idx == test_label(i)+1)
            num_correct = num_correct + 1;
        end
        
    end

% computing accuracy
accuracy = num_correct/test_size(1)






