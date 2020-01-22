# Spatial Embedding Loss
Tensorflow implementation of the Loss from ['Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth' by Neven et al.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Neven_Instance_Segmentation_by_Jointly_Optimizing_Spatial_Embeddings_and_Clustering_Bandwidth_CVPR_2019_paper.pdf)

This implementation uses the configuration with a 2-dimensional sigma and offset vectors pointing at the center of an instance in the embedding space, not in the image space. 
The loss takes in the embeddings (instead of the offset vectors), which can be calculated as follows:
```python
shape = tf.shape(offset) # 2 x h x w
res = [tf.range(shape[1]), tf.range(shape[2])]
dim_list = tf.meshgrid(*res, indexing='ij')
mesh = tf.stack(dim_list, axis=0) # 2 x h x w
# fill meshgrid with values between [0,1]
mesh = tf.divide(mesh, tf.reduce_max(mesh))
# tanh(offsets) turns them into unit vectors 
offset = tf.tanh(offset)
embeddings = mesh + offset # 2 x h x w
```

Dependencies: 
- Tensorflow 2.0
- Python 3.6.8  (or higher)
