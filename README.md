## Hello, I am Yoan Antonio 👋👦🏻
I am a software engineer pleased of working in the Semantic Web area. I am also a professor at the University of Informatics Science (UCI). I have taught Mathematics, Networks and Security, Programing. Currently, I am a Joint Ph.D. student (UCI-Ghent) and part of a team that is immersed in developing a **linked open data platform for the UCI**, so, if you like topics like semantic web, ontologies, linked data, knowledge graph embedding... WELCOME to our team!!!🍇

#### Linked university platform
What good does it do to have a linked open university platform? 
            
      -Representation and integration of data,
      -Reuse of data,
      -Linking data with related data out of the university,
      -Data level links provide an integration independent from application,
      -Building effective, integrated, and innovative applications over its datasets,
      -Revealing the contribution and achievements as LOD which is nowadays a mean to measure the university reputation.
      -University staff improvement.
      
             Nahhas, S., Bamasag, O., Khemakhem, M., Bajnaid, N.: 
             Added values of linked data in education: A survey and roadmap. 
             Computers 7(3), 45 (2018)
      
      
#### Linked university platform for The University of Informatics Science
-Raw data is collected from university repositories and databases and turned into linked datasets.

-Platform version 1 intends to build at least six datasets:
      
      -university staff
      -academic programs
      -internal places
      -scientific production (articles, papers, Ph.D. Thesis, MSc Thesis...)
      -productive and research projects
      -real-time streaming data (daily water and energy consumption)
      
#### The stages of the process of generating linked datasets
      -raw data collection,
      -defining the vocabulary model based on reusing existing ontologies and extend them when it is needed,
      -extracting and generating RDF datasets according to the defined vocabulary,
      -achieving interlinking among datasets internally and externally, 
      -storing the outcome datasets and exposing them via SPARQL endpoints, 
      -exploiting datasets by developing applications and services on top,
      -providing optimization and quality to output datasets.
      
             Nahhas, S., Bamasag, O., Khemakhem, M., Bajnaid, N.: 
             Added values of linked data in education: A survey and roadmap. 
             Computers 7(3), 45 (2018)
      

#### Universities have to face issues related to this process
      1-Lack of a unified, well-accepted vocabulary that satisfies all universities' requirements.
      2-The need of coping with the heterogeneity of datasets.
      3-The high cost of the existing SPARQL endpoint interfaces.
      4-Performance shortcomings of federated queries over current SPARQL endpoints.
      5-Incomplete data.
      
           -Nahhas, S., Bamasag, O., Khemakhem, M., Bajnaid, N.: 
            Added values of linked data in education: A survey and roadmap.
            Computers 7(3), 45 (2018)
           -Pereira, C.K., Siqueira, S.W.M., Nunes, B.P., Dietze, S.: 
            Linked data in education:a survey and a synthesis of actual research 
            and future challenges. IEEE Transactions on Learning Technologies 11(3), 400{412 (2017)
      
#### To cover previous issues in my Ph.D. research I 've separeted them into two works:
      1-Creating and exploiting linked datasets (issues from 1 to 4)
      2-Knowledge graphs completion (issue 5)
      
## The first work is a proof of concept about linked courses publication and consumption 
-This implementation intends to show an early application of the platform by easing data consumption to course recommender applications.
-Three components were developed:

  1- **rawdata_api** collects data from the university repositories and databases and arranges it into JSON files, one for each entity.
  
  -In this case, data about or related to university courses such as:
                        
       courses, academic terms, assessment methods, buildings, departments, faculties, languages, materials, 
       rooms, students, subjects, teachers, teaching methods, universities 
  
![imagen](https://user-images.githubusercontent.com/57901401/120075876-0b65f200-c071-11eb-8626-9e72aa5057e2.png)
              
         
 2-**coursesld_server** transforms raw data in JSON files (**rawdata_api** outputs) to courses files in RDF serializations such as JSON LD, TTL, N3,CSV.
 
 -It is a customized interface for publishing linked courses according to the client apps needs.
 
 ![imagen](https://user-images.githubusercontent.com/57901401/120075979-86c7a380-c071-11eb-84cf-04a08a4e3584.png)

 3-**coursesld_client** allows testing the **coursesld_server** interface. It is a proof of concept where different technologies allow automatizing the client-server communication such as Hydra/Tree vocabularies and the DCAT 2 metadata vocabulary.
 
 -It can request course fragments by the start date and the subject to multiple coursesld_server interfaces.
 
 ![imagen](https://user-images.githubusercontent.com/57901401/120078418-80d7bf80-c07d-11eb-9a83-247367bf071e.png)

## The second work is about knowledge graph completion
-We are trying to develop an algorithm for knowledge graph completion based on Bidirectional Encoder Representation from Transformers (BERT)

#### We have important examples of Knowledge graph
       Yago
       Wikidata
       DBpedia
       GDELT
       
#### knowledge graph are incomplete by nature
      Some of them are generated in automatic way
      They may have missing edges, they may not include all the fact 
      They work under open world assumption (OWA). Absence of a fact does not imply fact is false. Simply, 
      we do not know the fact. 
      
#### Machine Learning on knowledge graphs (Practical tasks)
   -Link prediction/Triple Classification (More famous one)
            
            -Knowledge graph completion
            -Content recommendation
            -Question answering
  -Collective node classification/Link-Based Clustering
          
            -Customer segmentation
  -Entity matching 
            
            -Duplicate detection
            -Inventory items deduplication
            
#### Link prediction/Triple Classification 
-Link prediction(We emphasize here)
            -Learning to rank problem
            -Information retrieval metrics
            -No ground truth negatives  in test set required
-Triple Classification
            -Binary Classification task
            -Binary classification metrics
            -Test set  requires positives and ground truth negatives

#### From feature engineering to graph representation learning
-feature engineering (Machine Learning)=> graph representation learning 
-graph representation learning (learning representation of nodes and edges automatically) (Deep Learning)
-We could use traditional deep learning tools
            -CNNs are designed for grids(e.g images)
            -RNNs/word2vec for sequences (e.g.text)
-But graphs are more complex:
=>We need ad-hoc models!
=>Graph representation learning
Learning representations of nodes and edges. We turn nodes and edges into vector representations
-Once we have vector representations we can carry out tasks as link prediction
-Handling vector representations is much better than handling nodes and edges. Vector can be processed by neural network architectures

#### Graph representation learning
Different solutions have been presented:
-Node Representation/Graph features based Methods (DeepWalk, node2vec)
-Graph Neural Networks (GNNs)
       -GCNs (similar to Knoledge Graph Embeddings), Graph Attention Networks
-Knowledge graph embeddings (KGE) (We focus on this method)
            TransE, DistMult, ComplEx, ConvE, ComplEx-N3, RotatE

#### Knowledge Graph Embeddings (KGE)
-Automatic, supervised learning of embeddings, i.e. projections of entities and relations into a continuos low-dimensional space.
-These embeddings normally have no more than a few hundred components
-These competing models try to achieve the same goal: locate the embeddings in a position able to maximize the chance of predicting 
missing links (unseen facts in the graph), catching symetry, asymmetry, inversion, composition relations. Also hierarchies, type constraints,
transitivity, homophily, long-range dependencies.



