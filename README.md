## Hello, I am Yoan Antonio 👋👦🏻
I am a software engineer pleased of working in the Semantic Web area. I am also a professor at the University of Informatics Science (UCI). I have taught Mathematics, Networks and Security, Programing. Currently, I am a Ph.D. student (Joint UCI-Ghent) and part of a team that is immersed in developing a **linked open data platform for the UCI**, so, if you like topics like semantic web, ontologies, linked data, knowledge graph embedding... WELCOME to our team!!!🍇

#### Linked university platform
What good does it do to have a linked open university platform? 
            
      -Representation and integration of data,
      -Reuse of data,
      -Building effective, integrated, and innovative applications over its datasets,
      -Revealing the contribution and achievements as LOD which is nowadays a mean to measure the university reputation.
      -University staff improvement.
      
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
      -exploiting datasets by developing applications and services on top, and, providing optimization and quality.

#### Universities have to face issues related to this process
      -Lack of a unified, well-accepted vocabulary that satisfies all universities' requirements.
      -The need of coping with the heterogeneity of datasets.
      -The high cost of the existing SPARQL endpoint interfaces.
      -Performance shortcomings of federated queries over current SPARQL endpoints.
      -Incomplete data.
      
#### In order to illustrate the architecture of the platform, a proof of concept about linked courses publication follows:
Three components were developed:

  1- **rawdata_api** collects data from the university repositories and databases and arranges it into JSON files, one for each entity. In this case, data about or related to university courses such as:
           
           courses, academic terms, assessment methods, buildings, departments, faculties, languages, materials, 
           rooms, students, subjects, teachers, teaching methods, universities 
           
 2-**coursesld_server** transforms raw data in JSON files (**rawdata_api** outputs) to courses files in RDF serializations such as JSON LD, TTL, N3,CSV. It is a customized interface for publishing linked courses according to the client apps needs.
 
 3-**coursesld_client** allows testing the **coursesld_server** interface. It is a proof of concept where different technologies allow automatizing the client-server communication such as Hydra/Tree vocabularies, the metadata vocabulary DCAT 2, etc.






