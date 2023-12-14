## Density Estimation Using Semantic Segmentation of Satellite Imagery Data via Federated Learning

This project focuses on developing a robust solution for accurately estimating building density in high-resolution satellite imagery through semantic segmentation, utilizing the Inria Aerial Image Labeling Dataset
To enhance privacy and computational efficiency, we integrate federated learning into the training pipeline, employing a client-server based environment consisting of 5 clients and a central server.
### Dataset
We are using the Inria Aerial Image Labeling Dataset. 

Dataset features:

- Coverage of 810 km² (405 km² for training and 405 km² for testing)
- Aerial orthorectified color imagery with a spatial resolution of 0.3 m
- Ground truth data for two semantic classes: *building* and *not building* (publicly disclosed only for the training subset)

The training set contains 180 color image tiles of size 5000×5000, covering a surface of 1500 m × 1500 m each (at a 30 cm resolution). There are 36 tiles for each of the following regions:

- Austin
- Chicago
- Kitsap County
- Western Tyrol
- Vienna

The format is GeoTIFF (TIFF with geo referencing, but the images can be used as any other TIFF). Files are named by a prefix associated to the region (e.g., Austin- or Vienna-) followed by the tile number (1-36). The reference data is in a different folder and the file names correspond exactly to those of the color images. In the case of the reference data, the tiles are single-channel images with values 255 for the *building* class and 0 for the *not building* class.
### Code
- All the models are defined in Models.py
- Models.py should be present in central server and all the client systems.
- Make sure to change the host ip address in fed_client.py to the central server's ip.
- All the clients should have fed_client.py along with train_[city].py
- Central server waits for atleast 3 clients to send their data (weights).

##### Central Server
```python3 fed_server.py```
##### Client
```python3 train_[city].py```
