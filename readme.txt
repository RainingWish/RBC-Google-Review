project discribtion:
  this project is for CUhackathon2020 RBC challenge. RBC wants to collect data from social media to get cudtomer feed back
  and analysis the data to improve RBC

steps
1. collect data using google place API
2. store raw data into elasticsearch 
3. process raw google review data and group them into topic using machine learning(NLP)    
    - model: LDA = gensim.models.ldamodel.LdaModel
4. put processed data into elasticsearch to another index
5. visualize the processed data using kibana dashboard

youtube demo: https://www.youtube.com/watch?v=zr7ESeG7H1I&t=1s 
