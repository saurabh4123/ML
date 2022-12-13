# ML Project
It's A Fruad .....

In this competition we are predicting whether an online transaction is fraudulent or not

Transaction Features:

TransactionDT: Timedelta from a given reference datetime (not an actual timestamp), but the time difference in seconds from a certain time.
TransactionAMT: Transaction payment amount in USD, the decimal part is worth paying attention to.
ProductCD: Product code, the product for each transaction. It may not necessarily be an actual product but may also refer to a service.
card1-card6: Payment card information, such as card type, card category, issue bank, country, etc.
addr1-addr2: Address, billing region and billing country
dist: Distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.
P_ and (R__) emaildomain: Purchaser and recipient email domain, some transactions do not require the recipient, and the corresponding Remaildomain is empty
C1-C14: Counting, such as how many addresses are found to be associated with the payment card, etc.
D1-D15: Timedelta, such as days between previous transaction, etc.
M1-M9: Match, such as names on card and address, etc.
Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations. Some V features are missing in different proportions.
Identity Features:

id_01-id_11: Numerical features for identity, which is collected by Vesta and security partners such as device rating, ip_domain rating, proxy rating, etc. Also it recorded behavioral fingerprint like account login times/failed to login times, how long an account stayed on the page, etc.
DeviceType, DeviceInfo and id_12-id_38: Categorical Features



                ##First of all download the config.txt file and open the terminal where file is downloaded
                Then type->
                pip install -r config.txt


Step 1-Open preprocessingdata.py present in the main branch.

Step 2-import the given(I/P) dataset by changing the path in the above file.

Step 3-Start doing EDA(exploratory data analysis) and preprocess the data as done in above file(you can make the required changes in your file if you want).

Step 4-Get the output of the above preprocesing in csv files using to_csv function as done in above file.

Step 5-Choose the model which you want to apply from the listed models.

Step 6-After choosing,open that model's python file and change the path to the new preprocessed data's path.

Step 7-Run the code and get the outputs.

Step 8-Produced O/Ps are the required results.

![image](https://user-images.githubusercontent.com/54600301/207419280-b24aa226-6fb2-443e-80ca-b0eb2f83bc0e.png)

