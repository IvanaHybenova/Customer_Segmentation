![image](https://user-images.githubusercontent.com/31499140/78498751-7c9b6500-774c-11ea-8713-680e3854913a.png)  

![image](https://user-images.githubusercontent.com/31499140/78500088-d1db7480-7754-11ea-8436-2304ed4da296.png)

![image](https://user-images.githubusercontent.com/31499140/78498829-ffbcbb00-774c-11ea-9c01-4f965b51977c.png)  

<details>
  <summary>Accompanying talk - click to expand!</summary>
  
  In the mall there are all sorts of decisions that have to be made. Among them there are some that directly effect the customer experience.
  It is for example the product selection, its organization in the mall, prices. So far, the only way we measure how good we are in doing these decisions is monthly revenue.
  But our mission is to build stable and happy customer base with great customer experience, which were among the main reasons of providing membership in the first place.  

  Since we don‘t know who our customers are, we currently have general marketing. Everybody are getting the very same emails and flyers about discounts or seasonal products in the mall etc.  
  Our marketing is not personalized in any way.

  Customer experience department is in close collaboration with marketing department and they are supposed to make sure that each of our customers finds in our store almost everything, 
  he/she might desire and that the whole experience in our store is as pleasant as possible. Without knowing our customer base, we treat every customer the same way.
  Now I am talking about the online questionnaires, content and frequency of the emails etc.  

  One of the main metrices we came up with to measure the „customer success“ was the spending score. We did our best to implement in the calculation all the features that might play
  a role in the customer behavior, so the customers with annual income 40K are not expected to spend as much as customers with annual income 80K etc.  

  We know that the spending score differs from customer to customer and the customer segmentation might help us to group the customers, understand which segments have low spending score
  and features of these segments might imply what could be possible improvements to make the customers satisfied and loyal.  
</details>  

![image](https://user-images.githubusercontent.com/31499140/78499406-ce45ee80-7750-11ea-9ef3-1e856a9a2f31.png)  

![image](https://user-images.githubusercontent.com/31499140/78499431-f59cbb80-7750-11ea-84df-d2533c0950f9.png)  

<details>
  <summary>Accompanying talk - click to expand!</summary>
  
   The Algorithm chosen for the clustering was very popular Kmeans clustering. The option Kmeans++ was chosen, to make sure that the clusters are initialized properly and most optimal clusters are found.
  The distance measure was chosen Euclidean distance, the most popular one. Euclidean distance measures the distance between two points with the shortest, straight line.

  On the picture we can see two points in two dimensional space – with X axis and Y axis. In our case we have 4 dimensions. To make sure that one axis is not too big compared to other I needed to 
  scale the data, so values for each feature are on the same scale, with mean 0 and standard deviation 1. With this approach I assured that each feature has the similar weight when creating the clusters.
  The effect of the standard scaling is clear from the sample.

  Besides that I want to point out that gender column taking values female and male, needed to be split into two columns - female and male with zeros and ones, since we need to use numbers
  In mathematical models. One simply means that the customers is a female and 0 that the customer is not a female (in Female column).
</details>  

![image](https://user-images.githubusercontent.com/31499140/78548119-7f579200-7800-11ea-8e0a-2cad0286fe7e.png)  

<details>
  <summary>Accompanying talk - click to expand!</summary>
  
  The Elbow Method is probably the most popular way to find out how many clusters are optimal for given dataset. WCSS means within clusters sum of squares, which basically tells us how far are the points
  clusters from their centroids. The point of this chart is to find the „elbow“ in the curve, which was in this case number 4. Creating 5 or more clusters would lead to higher similarity of points within each cluster, 
  but only a little.
</details>  

![image](https://user-images.githubusercontent.com/31499140/78548183-a2824180-7800-11ea-9e45-ad1e39fbeaf9.png)

<details>
  <summary>Accompanying talk - click to expand!</summary>
  
   It is difficult to visually check formed clusters with multiple dimensions, there it comes handy to use principal component analysis.
Without going into mathematical details – it is a statistical procedure that transforms the data from high level of multidimensional space 
into given number of dimensions space (2 in our case) without loosing any information. 
This is easy to plot and convenient to check our clusters -> how far (different) they are from each other.
</details>  

![image](https://user-images.githubusercontent.com/31499140/78499559-a6a35600-7751-11ea-865d-79daa8af84d6.png)  

[Found Segments](https://htmlpreview.github.io/?https://github.com/IvanaHybenova/Customer_Segmentation/blob/master/Customer_Segmentation-Results.html)

![image](https://user-images.githubusercontent.com/31499140/78499653-49f46b00-7752-11ea-9e01-416e8471a983.png)  
<details>
  <summary>Accompanying talk - click to expand!</summary>
  From the determined segments we can now understand where is the low spending score coming from.
  Since we have both man and women in the clusters with higher spending score, we can conclude that both genders find products of their interest
  and are willing to pay the prices we have – but there is still space for improvements!

  The same scenario applies to annual income. 

  And for the last – age. If we would just group the customers based on the spending score, we would see that the lower spending score have our older customers.
  We could suspect that people with higher annual income have broader interest of products, their living standard is „higher“, so they buy also things they don‘t necessarily need. 
  But not even these people decided to spend their extra money in our store!
  Why is that? What can we do to improve shopping experience of older customers?
</details>   

![image](https://user-images.githubusercontent.com/31499140/78499781-f1719d80-7752-11ea-98ce-f2c2f83e9c9b.png)  

<details>
  <summary>Accompanying talk - click to expand!</summary>
  From what we have learnt there are further questions, which investigation might lead us to proper solution.
  Well first of all – we saw that half of the customers willing to subscribe to the membership are older. They were willing to subscribe,
  even though they are apparently not that satisfied with the experience in our mall (narrow product selection, prices, orientation in the store, we don‘t know yet),
  and one would expect that older people are not willing to subscribe to everything. That implies that there is a huge potential in this area,
  regarding attraction of older customers and it will definitely pay off, to give them our attention.

  So, do we have products of their interest? There is a team of people that make decisions about our product selection, do they have
  older people in mind? Is there difference in their needs? Aren‘t some of these products often out of stock? 
  If we can‘t brainstorm anything obvious we can still do customer basket analysis, which we can maybe wait with until we explore other – quicker and less costly options.

  The team that suggests prices for our products definitely knows prices of our competitors. Can we double check that there are no 
  essential products that is overpriced?

  Next, let‘s face it, older people might find it harder to run back and forth in the mall getting the products as they come to their mind. Can we double check
  essential products are close to each other, the whole mall is reasonably organized and each section is properly labeled so it is easy to navigate?
  If we can‘t thing of anything obvious from the top of our heads, we can just do a survey asking 4 – 5 simple questions regarding navigation in the store, and we can actually
  ask whether they are missing some products – and guess what? We are going to save half of the costs and send the survey only to customers from segments 0 and 1 
  If the navigation is the problem and it is not possible or doesn‘t make sense to reorganize the mall anyway, we can quickly from the transactions determine „busy
  Hours“ for these customers and have 1-3 extra workers in the store making sure, that people searching for products get help.

  But the most inexpensive and quickest thing we can do right away is to actually talk to marketing and customer experience department about the marketing activities in the past months.
  Can we conclude that we were targeting mainly the younger audience which might have lead to this development we are discussing today?
</details>   

![image](https://user-images.githubusercontent.com/31499140/78499830-46151880-7753-11ea-9c9f-1bd5ae76313d.png)  

[Code to segment new customers](https://github.com/IvanaHybenova/Customer_Segmentation/blob/master/New_data_segments.ipynb)  
[The DAG file](https://github.com/IvanaHybenova/Customer_Segmentation/blob/master/customer_segments_DAG.py)  
[Code to perfrom cluster analysis](https://github.com/IvanaHybenova/Customer_Segmentation/blob/master/Customer_Segmentation.ipynb)
  
<details>
  <summary>Accompanying talk - click to expand!</summary>
  Having this segmentation analysis as a one time thing is not helpful in a long run, because in half a year we will be doing the same uninformed decisions as we are doing today.

  Therefore I wrote a script that gets the new customers and assign them a segment based on those we have created with these 200 customers. I scheduled it on our Airflow server to be triggered every night.

  It downloads the information from the customer table from our database and populate newly created „segment“ column. Both the model and the scaler is saved in the database as well, to make sure,
  that new customers are assigned to the segments properly. 

  The script for running the segmentation itself is not automated because decisions needs to be made during running the analysis (picking number of clusters, visually checking the clusters, checking whether the formed clusters make sense)
  The code is generic and any adjustments and extensions of the code (features in the analysis, logic of the functions as well) are easily applicable. And conducting analysis itself is very easy and straightforward.

  Additionally, each customer except „segment“ value has also value for another column „segment_origin“. It is „created“ label if the customer was part of the segment analysis,
  or „assigned“ in case of a new customer, where assigned segment was already existing, so it is easy to monitor the ratio of customers that were not part the segment creation.

  It is important, because we want to pick up any new segments that might form in the future, or that the features of each segments will slightly change.

  The change in the segments is especially important, because we want be able to assess whether our actions towards improving customer experience are fruitful.
</details>   

![image](https://user-images.githubusercontent.com/31499140/78499850-7f4d8880-7753-11ea-8c5a-2aff177c8a40.png)







