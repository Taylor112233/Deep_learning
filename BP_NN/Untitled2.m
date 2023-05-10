A = csvread('mnist_train_100.csv')     
n = 98;
%while n<11
    a = A(n,:);
    b = reshape(a,[28,28]);    
    b = rot90(b);
    figure(n)
    pcolor(b)
    colormap(gray)
    axis square
    n = n+1;
%end