# Domain-Generalization

DG: Leave One Weather Out Framework

Leave One Weather Out is a proposed learning strategy to improve the core problem of traditional DGs, which is a failure to generalize to an unseen environment that has never been seen before. Traditional DG methods learn by simply mixing several weathers, but the performance tends to drop significantly when unlearned weather emerges in the learning stage during real model testing.
In this study, as shown in Figure 8, each bad weather (fog, rain, snow, sand, night, and cloud) is set as an independent domain, learning with N-1 weather and excluding one weather for model testing.


<img width="414" height="304" alt="image" src="https://github.com/user-attachments/assets/26ba04b5-4561-4a23-ba35-d04615fc352d" />


This strategy complements the following problems that existing DGs have not solved.
1. Grouping by weather to learn by explicitly separating characteristic differences between weathers
2. Weather-specific balance sampling to avoid overfitting to specific weather
3. Experimentally validate the generalization performance of the model in never-before-seen weather


<img width="1060" height="674" alt="image" src="https://github.com/user-attachments/assets/0fdd673f-f95b-40ea-94a1-ea16ac3b1ea9" />


From the training data (clear weather) of the merge custom dataset MERGED_SD and the training data of MERGED_TD, data excluding one of each bad weather (fog, rain, snow, sand, night, and cloud) are sequentially merged to form a source domain, and each bad weather excluded from the training is configured as a target domain. In order to prevent overfitting of the hyperparameters of the model, the verification data is also constructed using the verification data of MERGED_SD and MERGED_TD in the same way as the training data. In this case, the training data and verification data are constructed by applying balanced sampling so that all weathers are evenly distributed. The balanced sampling rate of each dataset is applied based on the quantity of weather data with the least distribution. The model is designed to learn the Source Domain and evaluate how well it generalizes in an unknown environment that has never been seen as a Target Domain after verification [33]. YAML files, including training, verification, test path, and class information, are automatically generated in each iteration stage of bad weather weather to maintain consistency in dataset references and minimize manual errors. The YOLOv8l model is trained using the Source Domain and Target Domain images, excluding the selected bad weather. After learning, it is verified, and the model evaluates the performance and stores the visualization results on the test data corresponding to the excluded bad weather conditions, and repeats this process for all weather conditions to comprehensively analyze the domain generalization performance.
