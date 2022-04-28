
def cnn_layer_dim(input_size, kernel_size, strides, padding):
    output_size_x = ((input_size[0] - kernel_size[0] + 2*padding[0]) / strides[0]) + 1
    output_size_y = ((input_size[1] - kernel_size[1] + 2*padding[1]) / strides[1]) + 1
    return output_size_x, output_size_y

input_size = 2720, 3840
#2748-28

kernel_size=(10, 8)
padding = (0,0)
input_size = cnn_layer_dim(input_size, kernel_size, kernel_size, padding)
kernel_size=(4, 2)
input_size = cnn_layer_dim(input_size, kernel_size, kernel_size, padding)
kernel_size = (2, 5)
input_size = cnn_layer_dim(input_size, kernel_size, kernel_size, padding)
kernel_size=(2, 2)
input_size = cnn_layer_dim(input_size, kernel_size, kernel_size, padding)
#print(input_size)



input_size = 160,160
padding = (0,0)

kernel_size=(4,4)
stride = (4,4)
input_size = cnn_layer_dim(input_size, kernel_size, stride, padding)
print(input_size)

kernel_size=(2,2)
stride = (2,2)
input_size = cnn_layer_dim(input_size, kernel_size, stride, padding)
print(input_size)






