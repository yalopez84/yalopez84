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
-We are trying to develop an algorithm for knowledge graph completion based on Bidirectional Encoder Representation from Transformers (BERT) in the 
linked university context

#### We have important examples of Knowledge graph
       Yago
       Wikidata (FreeBase included)
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

#### Anatomy of a knowledge Graph Embedding Model
            -Knowledge graph (KG)G
            -Scoring function for a triple f(t)
            -Loss function L
            -Optimization algorithm
            -Negatives generation strategy

#### Translation-based Scoring Functions
            -TransE=||(Es + Rp)-Eo||n    //vectors addition
            -RotatE=-||EsoRp-Eo||n
#### Factorization-based Scoring Functions
            -Rescal:low-rank factorization with tensor product
            frescal=e^T*Wr*Eo
            -DistMult: bilinear diagonal model. Dot product (the problem here is with asymetric relations due to dot product is symetric)
            fdistmul=<Rp,Es, Eo>
            -ComplEx: complex embeddings(Hermitian dot product)
               i.e extends DistMult with dot product in C
             fcomplEx=Re(<rp,es,eo>)
#### Deeper Scoring Functions   //if you want to use other strategy
            -ConvE: reshaping +convolution
            -ConvKB: convolutions and dot product  (computationally expensive)
#### Other recent models
            -HolE
            -SimplE
            -QuatE
            -MurP
            -... Really there are a lot of scoring functions and scoring functions determine the method, so what we have to do is text with a method and see if it solves 
              our problem
#### Loss function  -Here, there are also a lot of methods
            -Pairwise Margin-Based Hinge Loss  It is possible to assign a bigger score to positive triple than negative triple by a margin ganma during the training stage
            -Negative Log-Likelihood/Cross Entropy
            -Binary Cross-Entropy, convE uses this loss function
            -Self adversarial, it introduces weight for the negative sample
            ...
            Many more: Multiclass Negative Log-likelihood, absolute Margin, etc
            //Tendency of papers is to present loss functions in addition to scoring functions
#### along loss function it is important to speak about regularizers
            -We have multiple ways to use regularizations
            -L1, L2
            -L3 (model complEx), leader model in this moment
            -Dropout (ConvE)
  Knowledge graph embedigns are into to select a scoring function, a loss function, a regularizer, etc. The idea is to combine part of the original models
#### Negatives Generation 
            -Due to we want the model to classify positive and negative facts, we have to train it with negative facts. Let us remember a knowledge graph only have truth fact but it also has missing facts.
            -To generate negative facts, we start from a locally closed world assumption, where we consider a triple is complete and changing part of that triple we create negative facts, example we have entities in the graphs {Mike, Liverpool, AcmeInc, George, LiverpoolFC}, relations {bornIn, friendWith} and the triple (Mike bornIn Liverpool).
            We generate as negative triples:
            -Mike bornIn AcmeInc
            -Mike bornIn LiverpoolFC
            -George bornIn Liverpool
            -AcmeInc bornIn Liverpool
            It is true that some negative are more negative than other. Here we can use a clever strategy to generate negative, but experiments have shown up this quite straghtforward tecnique works very well.
#### Strategies of training with Synthetic Negatives
            -Uniform sampling: generate all possible synthetic negatives and sample negatives for each positive 
            -Complete set: no sampling. Use all possible synthetic negatives for each positive t    
           -1-n scoring: batches of (s,p,*) or (*,p,o) labeled as positives (if included in training KG) or negatives (if not in training KG)
           Let us remember this aspect is peculiar in Knowledge Graph Embeddings because in other Machine Learning tasks you have positive and negative examples.
#### Training Procedure and Optimizer
            -the goal is minimizing the loss function
            -Optimizer learns optimal paramters(e.g embeddings). Off-the -shelf Stocastic Gradiant Descent variants (best results are obtained with AdaGrad, Adam)
                        -back propagation
                         -stocastic gradiant descent
            -Reciprocal triples 
                 -Injection of reciprocal triples in training set.e.g <Alice childOf Jack> <Jack childOf Alice> 
            *Previous aspects are important to stady models, to compare them and also to select frameworks to work.
#### Training ideas
            -As other Machine Learning models, here the aspects that we must have into account:
                        -hyperparameters
                        -tunning
                        -model selection
                        -size of the grid
                        -random search (models are selected in a random way)
#### How to messure the model success (we keep talking about link prediction)
            -The idea is to evaluate the probability the model gives to truth fact against to the probability the model gives to negative facts.
            -We use metrics that come from information retrieval:
                        -Mean rank (MR)
                        -Mean Reciprocal Rank (MRR)
                        -Hits@N
 #### Comparing SOTA Results is Tricky
            -Different training strategies (e.g synthetic negatives)
            -Reciprocal relations in training set?
            -Unfair or suboptimal hyperparameters selection
            -Evaluation protocol: how to behave with tie ranks?
            -Ablation studies!
            
#### Advanced Knowledge graph embeddings topics 
         -Calibration. Probabilities generated by KGE models are uncalibrated!
            For example, when we have a knowledge graph, it  does not have negative facts. Using caligration you can go testing in a way that the probability calibration is                    going plot like a diagonal. A model with the calibrated probabilties is better to clasify. The idea is using syntetic negative facts to obtain a probability more                clear and interpretable.
         -Multimodal Knowledge graph embeddings.
         -Temporal Knowledge Graphs
            Many real-world graphs represents timestamed conecpts.
            It is recommeded the KGs called ICEWS14, Yago15K, Wikidata
            -The idea here is designing models that leverage the time component in the predictions.
            -Between these models we have TTransE, TA-DistMult, ConT, TNTComplEx
            -TNTComplEx is the most recently. It has embeddings for each timestamp, order 4 tensor decomposition problem. ComplEx as descomposition method. It uses dot product
            -Let us remember that dot product (producto escalar) of two complex numbers is the real part: x=a+bi, y=c+di, dot product is ac+bd
          -Uncertain Knowledge graphs, each fact has a probability of being truth.
          -KGE and Neuro-Symbolic Reasoning. Rule-based models + KGE. The objective here is to use both: KGE strengths and rule-based interpretability
#### Open Research Questions
         -More expressive models
         -Support for multimodality (node and edge attributes, time awareness still in their infancy) 
         -Robustness and interpretability (techniques to dissect, investigate, explain and protect from adversarial attacks) 
         -better benchmarks. novel datasets.
         -beyond link prediction, multi path predictions, adoption in larger differentiable architectures to inject background knowledge from graphs.
         -neuro- symbolic integration. Integrating KGE non differentiable reasining regimes to get the best of different worlds.
#### Questions and answers session
         -tradeoff about loss functions: it is hard to say in advance what loss function will be the best in an specific case.
         -most used ones are: multiclasess, nLs. Depends on the model and the hyperparameters
         -Why not to use semantic rules to generate good negatives. The answer is that maybe that is a good idea but different models generate negatives as they were explian to           save time. 
#### Applications
         -Fharmaceutical industry (first of all, they say they got a knowledge graph about genes and other topics by integrating different datasets related to that. One they
          got the knowledge graph, they created embeddings in order to predict reations between nodes and know what genes are related to a target drug. So the task is to create the knowledge graph and then to predict relations, by the way, in this context there are million of nodes and links)
         -Human Resources
         -Products (product recomendations, relations between clients and customers)
         -Food and Beverages
#### Software ecosystem around knowledge graphs embeddings
         -OpenKE
         -AmpliGraph
         -PytorchBigGraph
         -Pykg2vec
         -LibKGE
      Comparison by taking into account features, scalability, sota reproduced, software development.
      Features (models, pre-training models, other features)
      If we see the frameworks looking which of them allows to carry out models like  TransE, DistMult, ComplEx, TransH, TransD, TransR, RESCAL, HolE, SimplE, ConvKB, etc.
      Most common models are TransE, DistMult, they are implemented in almost all frameworks.
#### Pre-trained models
      -There are framework that offer embedding pre-trained based on common KG as Wikidata, freebase, benchmark datasets. For example PyTorchBigGraph has a full pre-trained over wikidata. Let us remember, training a model requires time and skills.
#### Other features to assess KGE frameworks
            OpenKE, c++ implementation
            AmpliGraph, benchmarking AID and preprocessing
                        -formats RDF, csv, ntriples
                        -knowledge discovery API
                        -Visualization
                        -Model selection API
                        -Colab tutorial
           Pytorch Big Graoh
                        -high level operation
                        -scalability (particioning experimental GPU)
                        
#### Scalability
           -When we need to train the model we need a scalable framework
           -All framework developed are based on Pytorch or TensorFlow. Most of them are based on Pytorch. OpenKE and PyKG2vec are based on both Pytorch and TensorFlow.
           -All framework suppor large KG, for example AmpliGraph supports graphs of 10^8 edges and 4*10^7 nodes.
       
#### Which library should I use?
            -take into account your expirience
            -the time you have to learn
            -The task to be solved
            -Frameworks the ibrary supports Pytorch, TensorFlow
            -accuracy, maturity
#### My Proposal version1
-Revision del estado del arte de los principales modelos de embedings en los dataset universitarios con enfasis en la generacion de negatices facts. El aporte de la publicacion estara dado en la generacion de hechos negativos basados en la semantica de las ontologias que subyacen. Se explican las estrategias de generacion de hechos negativos actuales a partir de las tripletas existentes. El problema de lograr buenos negativos basados en reglas semanticas esta en que consume mucho tiempo pero y que tal si tenemos un api que dado un dataset y su esquema ontologico nos logra generar un conjunto de tripletas negativas.
-Otra propuesta pudiera ser a partir del 'exito que han tenido la utilizaci'on de los modelos de embeddings de palabras como BERT revisar la aplicacion de los otros modelos en los grafos RDF. Se tienen word2vec (implementaci'on mas conocida), GloVe, BERT (relaciones de dependencia gramatical en oraciones), FastText (aspectos morfologicos de las palabras). Aqui he visto los trabajos con BERT pero una propuesta seria la utilizacion de todos estos algoritmos con un mismo pre entrenamiento para ver cual predice mejor en un dataset dado de aplicaci'on.
-Tener en cuenta como un aspecto negativo principalmente a la segunda propuesta es no contar con grandes bases de entrenamiento en otros idiomas ademas de Ingles, necesarios en los modelos de embeddins de palabras.
-Siguiendo la idea de lo que ocurre en el procesamiento del lenguaje natural. Se consideran 3 conceptos principales: el corpus de textos que sirven de fuente, el algoritmo para crear los word embeddings y el contexto utilizado por ese algoritmo. Los algoritmos se pueden dividir en dos grupos: metodos basados en contar cuantas veces aparece la palabra vecina de otras en el corpus, metodos que predicen una palabra a partir de sus vecinas y al mismo tiempo crean un vector de pesos que puede predecir el contexto en que se encuentra la palabra. Algoritmos principales para crear representaciones de palabras son Tf-idf, GloVe, Word2Vec. En la tesis que analizo se uso word2vec y los componentes principales son el corpus, el tama;o de la ventana y algoritmo word2vec.
-Se establece el tama;o de la ventana como una constante c que cuenta las palabras a la derecha y a la izquierda de la palabra objetivo. El contexto de cada palabra tiene un tama;o de 2c y abarca todo el tama;o de la ventana.con esos contextos por palabra es posible entrenar un modelo de prediccion con el que se construyen esos vectores. Vectores indican posiciones de palabras y por tanto dos palabras con similitud de vectores pudieran ser antonimos como es el caso de bueno y malo ya que ocupan posiciones similares en las oraciones.

-La representacion de palabras en un espacio semantico, comunmente llamada en la literatura como word embedding consiste en una representacion vectorial, que mediante
un modelado algebraico, pretende representar el signicado de una palabra.

-Word2vec en lugar de contar la cantidad de veces que una palabra esta en el contexto 
La idea por detras de word2vec es entrenar el algoritmo a partir de ir viendo si una palabra c esta en el contexto de la palabra w. Tiene dos implementaciones CBOW (conituos bag of words), Skip-grama. Ambas tratan de predecir si una palabra c esta en el contexto de una palabra objetivo w

-Pregunta de investigacion, analizar cual de los algoritmos de aprendizaje de representacion de grafos ya sea los tradicionales (TransE, etc) o los algortimos de representacion de palabras (word2vect, fasttext, Bert) adaptados a los grafos en el dominio de los datasets vinculados universitarios ofrece mejores resultados. Se propone el entrenamiento de los algoritmos en los dataset de caracter general FB y en los dataset de contextos especificos como los de Open university, Obuda y Southampton. Finalmente se concluye con su aplicacion en el contexto de la universidad de CI. Problema aqui es que entiendo que los algoritmos en el caso de PLN si son para aprender representaciones de palabras pero en el caso de grafos tienen mas bien que ver  

-Partimos de saber en la practica como se obtienen los KGE, se tienen tecnicas que encuentran los embedings solo a partir del conocimiento del grafo como TransE, etc. Aqui la idea es obtener vectores para nodos y relaciones de acuerdo a evaluar con operaciones entre vectores e ir mejorando esos vectores. Una manera de verlo claro seria tener un vector inicial para cada nodo y relacion del grafo, ir pasando por el algoritmo todas las tripletas, cada vez que pase 

-en la generacion de negatives fact aunque se tienen los enfoques de OWA y CWA, se dice que CWA trae desventajas, en el caso de OWA se tienen las estrategias de generar negativos por cada positivo (cambiando cabeza o cola o relacion) y otro enfoque es generar negativos teniendo en cuenta que los nuevos valores esten presentes en otras tripletas en el grafo. Aqui me pudiera plantear avanzar en la generacion de negatives fact desde la estrategia de OWA donde se evaluen semanticamente los nuevos valores pudiendo generar buenos negativos.

#### Hands on session
2:44
         

            
           
