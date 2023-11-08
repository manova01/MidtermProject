# About Dataset
## Context
An international e-commerce company based wants to discover key insights from their customer database. They want to use some of the most advanced machine learning techniques to study their customers. The company sells electronic products.

## Content
The dataset used for model building contained 10999 observations of 12 variables.
The data contains the following information:

ID: ID Number of Customers.

Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.

Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.

Customer care calls: The number of calls made from enquiry for enquiry of the shipment.

Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).

Cost of the product: Cost of the Product in US Dollars.
Prior purchases: The Number of Prior Purchase.

Product importance: The company has categorized the product in the various parameter such as low, medium, high.

Gender: Male and Female.
Discount offered: Discount offered on that specific product.


Weight in gms: It is the weight in grams.

Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.
## Objective
predict if  Product Shipment Delivered on time or not? To Meet E-Commerce Customer Demand
## How It Works 

Everything here runs locally. If you want to try out the service, follow the steps below:

Before you proceed, create a virtual environment.

I used python version 3.10. To create an environment with that version of python using Conda:

conda create -n <env-name> python=3.10

Just replace <env-name> with any title you want. Next:

 conda activate <env-name>

to activate the environment.


## Dockerfile
Now run:

 pip install -r requirements.txt

to install all necessary external dependencies.

Next, Run:

docker build -t <service-name>:v1 .

Replace <service-name> with whatever name you wish to give to the body fat percent estimator service, to build the image.

To run this service:

docker run -it --rm -p 9696:9696 <service-name>:latest

NOTE: I am running this on Windows hence Waitress. If your local machine requires Gunicorn, I think the Dockerfile should be edited with something like this:


RUN pip install -U pip

WORKDIR /app

COPY [ "online_webservice_flask/predict.py", "models/pipeline.bin", "requirements.txt", "./" ]

RUN pip install -r requirements.txt

EXPOSE 9696 
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

If the container is up and running, open up a new terminal. Reactivate the Conda environment. Run:

python predict.py

NOTE: predict.py is an example of data you can send to the ENTRYPOINT to interact with the service. Edit it as much as you desire and try out some predictions.
