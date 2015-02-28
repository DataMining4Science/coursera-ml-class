function [V]  = toBinaryVectorMatrix(v, num_labels)

m = size(v, 1);

V = zeros(m, num_labels);

for i = 1:m
  V(i,:) = (1:num_labels == v(i));
end

end
