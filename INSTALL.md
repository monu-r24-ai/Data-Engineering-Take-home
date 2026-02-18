# Installation Instructions
 

**IMPORTANT**: *Any installation requires at least 8Gb of RAM for proper operation.*

1. Install Docker Desktop

	  **Using Windows**: Install docker desktop https://docs.docker.com/desktop/install/windows-install/
	
	  **Using MacOS**: Follow instructions in the below link to setup your MacOS https://arjon.es/2019/setting-up-macbook-pro-for-development/

2. Ensure Docker Desktop is up and running by following below validations:
	
	Open Docker desktop and check for **my_assesment** project having 5 containers as below

	  ![docker-desktop](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/pictures/docker-desktop-containers.png)

3. Spin-up Spark cluster: 
	a. Open shell and run the below lines
	b. Change directory to the /project1
		cd {project1-directory}
	c. Docker compose
	
		docker compose -f ./docker-compose.yml --project-name my_assesment up
	
	
   	Validate if Spark Cluster is up and running in the Spark Master UI with 2 worker nodes -
	
   	![spark-master-ui](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/pictures/spark-master-ui.png)

