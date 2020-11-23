from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image
from pylab import *
name = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
def max_activation(layers_to_visualize, filters_to_visualize):
    model = load_model('model/cnn.h5')
    # Swap softmax with linear
    ## We also swap the Softmax activation function in our trained model,
    # which is common for multiclass classification problems, with the linear activation function.
    # Why this is necessary can be seen in the images below. Since you’re essentially looking backwards,
    # from outputs and fixed weights to inputs, you need a free path from outputs back to inputs.
    # Softmax disturbs this free path by essentially transforming your model data in intricate ways,
    # which makes the activation maximizations no longer understandable to humans.
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    # Visualize
    for layer_index in layers_to_visualize:
        for filter_index in filters_to_visualize:
            visualization = visualize_activation(model, layer_idx=layer_index,
                                                 filter_indices=filter_index)  # 使得第layer_idx层的第filter_indices个filter值最大
            # plt.subplot(len(filters_to_visualize)//8 + 1, 8, filter_index + 1)
            plt.imshow(visualization)
            plt.xticks([])
            plt.yticks([])
            if layer_index == -1:
                plt.title(name[filter_index])
                # 去除图像周围的白边
                plt.axis('off')
                height, width, channels = visualization.shape
                plt.gcf().set_size_inches(width / 100, height / 100)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(f'img/class{filter_index}-output-max-activation')
            else:
                plt.savefig(f'img/layer{layer_index}-max-activation')


def main():
    # max_activation(layers_to_visualize=[1, 5, 9], filters_to_visualize=range(64))
    max_activation(layers_to_visualize=[-1], filters_to_visualize=range(7))

    model = load_model('model/cnn.h5')
    for i in range(7):
        # 读取图片并转为数组
        im = Image.open(f'img/class{i}-output-max-activation.png')
        im = im.convert('L')
        im = array(im)
        im = im[np.newaxis,:,:,np.newaxis]

        pred = model.predict(im)
        cls = np.argmax(pred, axis=1)

        print(f'cls:{cls}  probability:{pred[0][cls]} ')

if __name__ == '__main__':
    main()
