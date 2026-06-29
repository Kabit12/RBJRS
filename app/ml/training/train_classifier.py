"""
Resume Classifier Training Script
====================================
Trains the TF-IDF + SVM resume classifier on sample data.

Since the Kaggle dataset requires manual download, this script provides:
1. A built-in sample dataset for immediate training (150+ samples, 25 categories)
2. Support for loading external CSV datasets
3. Model evaluation with accuracy, precision, recall, F1 metrics
4. Saved model files in ml_models/ directory

Usage:
    python -m app.ml.training.train_classifier
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.ml.text_preprocessor import text_preprocessor
from app.ml.resume_classifier import ResumeClassifier


# Built-in sample training data — representative resume snippets for each category
# In production, this would be replaced with the full Kaggle dataset
SAMPLE_DATA = {
    'Data Science': [
        'python machine learning data analysis pandas numpy scikit-learn tensorflow deep learning neural networks statistics regression classification clustering nlp computer vision data visualization matplotlib seaborn jupyter notebook sql big data',
        'data scientist with expertise in machine learning algorithms statistical modeling predictive analytics python r sql tableau power bi data mining feature engineering model deployment aws sagemaker spark hadoop',
        'phd in statistics with focus on bayesian methods developed deep learning models for image classification natural language processing sentiment analysis recommendation systems ab testing experimental design',
        'experienced in building end-to-end ml pipelines using python scikit-learn pytorch tensorflow data preprocessing feature engineering model training evaluation deployment using docker kubernetes mlflow',
        'data science professional skilled in python r sql statistical analysis machine learning deep learning natural language processing computer vision time series forecasting anomaly detection recommender systems',
        'built predictive models using random forest gradient boosting xgboost neural networks for customer churn prediction fraud detection demand forecasting using large datasets big data technologies',
    ],
    'Python Developer': [
        'python developer experienced in django flask fastapi rest api development postgresql mongodb redis celery docker kubernetes aws microservices architecture test driven development ci cd github actions',
        'senior python engineer building scalable web applications using django rest framework sqlalchemy alembic redis celery docker compose nginx unit testing pytest integration testing performance optimization',
        'python backend developer with expertise in asyncio aiohttp fastapi websockets graphql postgresql elasticsearch rabbitmq kafka event driven architecture microservices design patterns solid principles',
        'full stack python developer experienced in django react postgresql docker aws lambda serverless architecture ci cd pipelines code review mentoring agile scrum methodology',
        'python software engineer specializing in api development automation scripting data processing etl pipelines web scraping selenium beautiful soup requests flask sqlalchemy pytest git linux',
        'experienced python developer proficient in object oriented programming design patterns flask django fastapi rest apis database management sql nosql docker containerization cloud deployment',
    ],
    'Java Developer': [
        'java developer spring boot microservices rest api hibernate jpa postgresql docker kubernetes maven gradle junit mockito ci cd jenkins git agile scrum design patterns solid principles',
        'senior java engineer experienced in enterprise applications spring framework spring cloud netflix oss kafka rabbitmq elasticsearch docker kubernetes aws ec2 s3 rds microservices architecture',
        'java full stack developer with expertise in spring boot angular react typescript postgresql mongodb redis docker kubernetes jenkins ci cd aws azure devops practices',
        'java software engineer proficient in core java collections multithreading concurrency spring boot hibernate rest apis sql databases microservices design patterns test driven development',
        'experienced java developer building high performance distributed systems using spring boot kafka redis postgresql docker kubernetes aws lambda serverless architecture event sourcing cqrs',
        'java backend developer skilled in spring framework hibernate jpa microservices rest apis sql nosql databases docker containerization kubernetes orchestration ci cd pipelines agile methodology',
    ],
    'Web Designing': [
        'web designer ui ux design figma sketch adobe photoshop illustrator html css javascript responsive design mobile first design wireframing prototyping user research accessibility wcag',
        'frontend developer specializing in html css javascript react vue angular responsive web design css grid flexbox sass bootstrap tailwind material design figma adobe xd prototype',
        'creative web designer with strong portfolio in user interface design user experience research wireframing mockups html css javascript wordpress shopify e-commerce web accessibility',
        'ux ui designer skilled in figma adobe xd sketch user research personas journey mapping wireframing prototyping usability testing design systems component libraries responsive design',
        'web design professional experienced in creating modern responsive websites using html5 css3 javascript jquery bootstrap material design adobe creative suite figma interaction design',
        'frontend web designer proficient in html css sass less javascript typescript react angular vue responsive design cross browser compatibility web accessibility performance optimization',
    ],
    'HR': [
        'human resources manager recruitment talent acquisition employee relations performance management compensation benefits payroll hris workday sap hr analytics organizational development training leadership',
        'hr professional experienced in full cycle recruiting onboarding employee engagement performance reviews succession planning compliance labor law diversity inclusion training development hris systems',
        'talent acquisition specialist skilled in sourcing screening interviewing job posting employer branding linkedin recruiting applicant tracking systems workday greenhouse lever bamboohr',
        'hr business partner supporting organizational change management employee relations labor relations performance management compensation analysis benefits administration compliance regulatory requirements',
        'human resources generalist with experience in recruitment onboarding employee relations payroll administration benefits management training development policy implementation hr information systems',
        'senior hr manager leading recruitment retention strategies employee engagement programs performance management systems compensation benchmarking organizational development workforce planning analytics',
    ],
    'DevOps Engineer': [
        'devops engineer experienced in ci cd pipelines jenkins github actions docker kubernetes terraform ansible aws azure monitoring prometheus grafana linux shell scripting python automation infrastructure as code',
        'site reliability engineer skilled in kubernetes docker helm terraform aws gcp monitoring alerting incident response capacity planning performance tuning linux networking ci cd automation',
        'devops professional proficient in cloud infrastructure aws azure gcp docker containerization kubernetes orchestration terraform infrastructure as code ansible configuration management jenkins ci cd pipeline automation',
        'cloud devops engineer building and maintaining ci cd pipelines using jenkins github actions docker kubernetes terraform aws services ec2 ecs eks rds s3 cloudformation monitoring logging',
        'devops specialist experienced in automating deployment pipelines infrastructure provisioning configuration management monitoring using docker kubernetes terraform ansible prometheus grafana elk stack',
        'senior devops engineer designing scalable cloud architectures aws azure implementing ci cd pipelines docker kubernetes terraform ansible monitoring observability security compliance automation',
    ],
    'Business Analyst': [
        'business analyst experienced in requirements gathering stakeholder management process improvement data analysis sql tableau power bi jira confluence agile scrum methodology documentation uml use cases',
        'senior business analyst skilled in business process modeling requirements documentation gap analysis data analysis reporting sql excel power bi stakeholder communication agile waterfall methodology',
        'business systems analyst with expertise in requirements elicitation process mapping user stories acceptance criteria testing coordination stakeholder management jira confluence sql data analysis',
        'business analyst professional experienced in functional specifications process improvement data driven decision making sql database queries reporting dashboards power bi tableau agile scrum',
        'it business analyst bridging gap between business and technology requirements analysis use case development process optimization data analysis sql reporting tools project coordination',
        'business analyst with experience in financial services requirements gathering analysis documentation stakeholder management process improvement data analysis reporting agile methodology scrum',
    ],
    'Mechanical Engineer': [
        'mechanical engineer experienced in cad solidworks autocad fea analysis thermal design product development manufacturing processes materials science 3d printing cnc machining quality control iso standards',
        'senior mechanical design engineer proficient in solidworks catia ansys fea cfd analysis product lifecycle management geometric dimensioning tolerancing manufacturing optimization quality assurance',
        'mechanical engineer skilled in automotive design hvac systems fluid mechanics thermodynamics product development prototyping testing validation solidworks autocad matlab simulink project management',
        'mechanical engineering professional with expertise in machine design manufacturing engineering quality control six sigma lean manufacturing solidworks autocad finite element analysis',
        'product design mechanical engineer experienced in cad modeling solidworks creo simulation analysis dfm dfa prototyping 3d printing injection molding sheet metal design testing',
        'mechanical engineer specializing in thermal systems heat transfer fluid dynamics computational fluid dynamics ansys fluent solidworks product development manufacturing optimization',
    ],
    'Sales': [
        'sales professional experienced in b2b sales account management lead generation crm salesforce hubspot negotiation closing pipeline management client relationship revenue growth strategic planning',
        'senior sales executive skilled in enterprise sales solution selling consultative approach account management key accounts revenue targets pipeline management crm salesforce business development',
        'sales manager leading team of representatives territory management quota achievement client acquisition retention strategic planning market analysis competitor research presentation skills',
        'business development representative experienced in cold calling email outreach lead qualification appointment setting crm management pipeline building sales presentations product demonstrations',
        'sales professional with track record of exceeding targets account management customer relationship building negotiation presentation skills salesforce crm data driven sales strategies',
        'inside sales representative skilled in lead generation qualification b2b sales pipeline management crm tools salesforce hubspot phone sales email marketing territory management',
    ],
    'Civil Engineer': [
        'civil engineer experienced in structural design autocad revit etabs safe analysis construction management project planning quantity surveying building codes seismic design concrete steel structures',
        'structural civil engineer proficient in design of reinforced concrete steel structures foundation design retaining walls bridges using etabs staad pro autocad revit project management',
        'civil engineering professional skilled in highway design water resources environmental engineering surveying geotechnical analysis construction supervision project management autocad civil 3d',
        'civil engineer with expertise in structural analysis design construction management building codes compliance autocad revit staad pro quantity estimation cost control quality management',
        'senior civil engineer specializing in infrastructure projects road design drainage systems structural design autocad revit project management construction supervision contract administration',
        'civil engineer experienced in transportation engineering highway design traffic analysis environmental impact assessment surveying gis autocad civil 3d project coordination',
    ],
    'Advocate': [
        'advocate practicing in civil criminal corporate law litigation arbitration mediation legal research drafting contracts legal documentation compliance regulatory affairs client counseling court appearances',
        'lawyer experienced in corporate law contract drafting legal compliance intellectual property trademark copyright mergers acquisitions due diligence legal advisory dispute resolution',
        'legal professional specialized in criminal law civil litigation family law property law labor law legal research case analysis court representation client counseling documentation',
        'advocate with expertise in constitutional law human rights law public interest litigation legal research writing appeals court proceedings client representation documentation',
        'senior advocate practicing civil commercial law arbitration mediation contract negotiations corporate governance compliance regulatory affairs legal advisory dispute resolution',
        'legal counsel experienced in corporate commercial law contract drafting review compliance advisory intellectual property employment law litigation risk management legal documentation',
    ],
    'Arts': [
        'artist graphic designer creative professional skilled in digital illustration traditional drawing painting typography color theory adobe photoshop illustrator indesign procreate art direction',
        'creative arts professional experienced in visual design animation motion graphics video editing photography adobe creative suite after effects premiere pro photoshop illustrator branding',
        'fine arts graduate with expertise in painting sculpture mixed media installation art gallery exhibitions portfolio development art history contemporary art creative thinking visual storytelling',
        'multimedia artist skilled in digital art animation 3d modeling graphic design typography color theory composition adobe creative suite blender maya creative direction branding',
        'art director with experience in creative campaigns branding visual identity design print digital media photography illustration typography adobe creative suite team leadership',
        'creative professional specialized in illustration digital art graphic design animation video production photography adobe creative suite procreate content creation visual storytelling',
    ],
    'Automation Testing': [
        'automation test engineer experienced in selenium webdriver java python testng junit cucumber bdd api testing rest assured postman ci cd jenkins performance testing jmeter mobile testing appium',
        'qa automation engineer skilled in test framework development selenium java python robot framework pytest cypress api testing database testing ci cd pipeline integration test strategy',
        'senior automation tester proficient in selenium webdriver java cucumber bdd test automation framework design api testing rest assured postman jira agile scrum test planning execution',
        'software test automation engineer experienced in selenium java testng maven jenkins ci cd api testing performance testing jmeter mobile testing appium agile methodology defect management',
        'test automation specialist skilled in building test frameworks selenium python pytest robot framework api testing postman rest assured database testing ci cd jenkins docker',
        'quality assurance automation engineer experienced in selenium java python cucumber bdd tdd test strategy test plan execution regression testing performance testing ci cd pipelines',
    ],
    'Blockchain': [
        'blockchain developer experienced in solidity ethereum smart contracts web3 truffle hardhat defi nft ipfs consensus mechanisms distributed ledger technology cryptocurrency tokenomics',
        'blockchain engineer skilled in solidity rust ethereum polygon smart contract development web3.js ethers.js defi protocols nft marketplace development token standards erc20 erc721',
        'cryptocurrency blockchain developer proficient in smart contract auditing solidity ethereum layer 2 solutions defi yield farming liquidity protocols dao governance token economics',
        'blockchain architect experienced in designing distributed systems ethereum hyperledger consensus algorithms cryptography smart contracts dapp development web3 integration decentralized finance',
        'web3 developer skilled in solidity javascript typescript react ethereum smart contracts defi protocols nft development ipfs decentralized storage blockchain security audit',
        'blockchain developer with expertise in smart contract development ethereum solidity web3 technologies decentralized applications cryptocurrency tokenization defi protocols security auditing',
    ],
    'Database': [
        'database administrator experienced in mysql postgresql oracle sql server mongodb database design optimization performance tuning backup recovery replication high availability clustering stored procedures',
        'senior dba skilled in database architecture design optimization performance tuning mysql postgresql oracle sql server nosql mongodb redis data modeling etl processes backup disaster recovery',
        'database engineer proficient in sql server administration performance tuning query optimization indexing stored procedures triggers views database security audit compliance data migration',
        'database developer experienced in sql plsql database design normalization stored procedures functions triggers performance optimization mysql postgresql oracle data warehousing etl',
        'data engineer skilled in database management mysql postgresql mongodb redis elasticsearch data pipeline design etl processes data modeling sql query optimization cloud databases aws rds',
        'database administrator with expertise in oracle mysql postgresql database design performance tuning backup recovery replication clustering high availability security compliance',
    ],
    'DotNet Developer': [
        'dotnet developer experienced in c# asp.net core mvc web api entity framework sql server azure microservices docker kubernetes rest api design patterns solid principles ci cd',
        'senior .net engineer skilled in c# asp.net core blazor entity framework core sql server azure services microservices architecture docker kubernetes ci cd azure devops',
        'full stack .net developer proficient in c# asp.net mvc web api angular react sql server entity framework azure cloud services microservices docker unit testing',
        '.net software engineer experienced in c# asp.net core web api entity framework sql server redis azure functions microservices docker kubernetes design patterns tdd',
        'c# .net developer building enterprise applications using asp.net core entity framework sql server azure microservices docker kubernetes ci cd pipelines unit testing integration testing',
        'dotnet developer skilled in c# asp.net mvc core web api entity framework sql server azure cloud services rest api development microservices architecture testing deployment',
    ],
    'Electrical Engineering': [
        'electrical engineer experienced in power systems circuit design pcb layout embedded systems microcontrollers plc programming scada automation control systems matlab simulink autocad electrical',
        'senior electrical engineer skilled in power distribution protection systems switchgear transformers circuit breakers relay coordination electrical design autocad etap power system analysis',
        'electrical engineering professional proficient in embedded systems arm cortex microcontrollers fpga vhdl verilog pcb design altium eagle circuit simulation spice signal processing',
        'electrical engineer with expertise in industrial automation plc programming scada hmi drives motors variable frequency drives control systems instrumentation electrical design standards',
        'power systems electrical engineer experienced in substation design transmission distribution protection coordination power factor correction harmonic analysis load flow short circuit analysis etap',
        'electrical engineer specializing in renewable energy solar pv wind power battery storage systems power electronics inverter design grid integration electrical design autocad',
    ],
    'ETL Developer': [
        'etl developer experienced in informatica talend ssis data warehousing sql oracle data pipeline design extraction transformation loading data quality data governance business intelligence',
        'data integration specialist skilled in etl processes informatica powercenter talend ssis sql server oracle data warehouse design dimensional modeling star schema snowflake schema',
        'senior etl developer proficient in building data pipelines informatica talend python sql data warehousing data lake architecture aws redshift snowflake data quality validation',
        'etl data engineer experienced in informatica powercenter ssis talend python sql data warehousing oracle sql server performance tuning data validation error handling scheduling',
        'data warehouse developer skilled in etl design informatica ssis sql stored procedures dimensional modeling data marts olap cubes reporting analytics business intelligence tools',
        'etl developer with expertise in data pipeline development informatica talend python sql database management data quality transformation loading scheduling monitoring optimization',
    ],
    'Hadoop': [
        'hadoop big data engineer experienced in hdfs mapreduce hive pig spark scala java yarn cluster management data processing data pipeline etl sqoop flume kafka',
        'big data developer skilled in hadoop ecosystem hive spark scala python data lake architecture aws emr s3 kafka streaming real time processing data engineering',
        'hadoop administrator experienced in cluster management hdfs yarn hive spark ambari cloudera hortonworks data pipeline design performance tuning security kerberos monitoring',
        'big data engineer proficient in hadoop spark scala hive pig sqoop kafka data pipeline development data lake architecture cloud platforms aws azure gcp',
        'data engineer specializing in big data technologies hadoop spark hive kafka flink stream processing batch processing data warehousing data lake etl pipelines',
        'hadoop developer experienced in big data ecosystem hdfs mapreduce hive spark scala java python data processing analytics pipeline development cluster management',
    ],
    'Health and Fitness': [
        'health fitness professional certified personal trainer nutrition specialist exercise science kinesiology weight management cardio strength training flexibility mobility wellness coaching group fitness',
        'fitness trainer experienced in personal training group classes exercise programming nutrition counseling weight loss body composition strength conditioning flexibility mobility wellness coaching certification',
        'health and wellness professional skilled in fitness assessment exercise prescription nutrition planning wellness programs corporate wellness stress management health education lifestyle coaching',
        'certified fitness instructor experienced in group exercise personal training nutrition guidance weight management cardiovascular conditioning strength training yoga pilates functional fitness',
        'health fitness specialist with expertise in exercise physiology sports nutrition personal training program design body composition analysis fitness testing health promotion wellness coaching',
        'personal fitness trainer certified in strength conditioning nutrition exercise science specializing in weight management muscle building cardiovascular fitness flexibility functional movement',
    ],
    'Network Security Engineer': [
        'network security engineer experienced in firewall configuration ids ips vpn siem penetration testing vulnerability assessment compliance iso27001 pci dss network monitoring cisco palo alto',
        'cybersecurity professional skilled in network security firewall management intrusion detection prevention siem splunk qradar vulnerability management penetration testing incident response threat analysis',
        'information security engineer proficient in network architecture security design firewall configuration cisco asa palo alto checkpoint ids ips vpn ssl encryption compliance audit',
        'security analyst experienced in siem monitoring incident response threat hunting vulnerability assessment penetration testing network security firewall management compliance frameworks iso nist',
        'network security specialist skilled in cybersecurity threat detection incident response firewall administration ids ips vpn siem tools security audit compliance regulatory frameworks',
        'senior security engineer experienced in network security architecture firewall management penetration testing vulnerability assessment incident response siem monitoring compliance certification cissp ceh',
    ],
    'Operations Manager': [
        'operations manager experienced in process optimization supply chain management logistics warehouse management inventory control quality assurance lean manufacturing six sigma project management team leadership',
        'senior operations professional skilled in process improvement supply chain optimization logistics management team leadership budgeting resource allocation kpi performance monitoring strategic planning',
        'operations manager with expertise in manufacturing operations production planning quality control inventory management supply chain logistics lean six sigma continuous improvement team management',
        'business operations manager experienced in process re-engineering workflow optimization team management budgeting forecasting vendor management stakeholder communication strategic planning execution',
        'operations director skilled in operational strategy supply chain management logistics process improvement lean six sigma quality management team leadership resource planning performance metrics',
        'operations manager specializing in process optimization efficiency improvement team leadership project management supply chain logistics inventory management quality assurance customer satisfaction',
    ],
    'PMO': [
        'project management professional pmp certified experienced in project planning execution monitoring controlling risk management stakeholder management ms project jira agile waterfall hybrid methodology',
        'program manager pmo lead skilled in portfolio management project governance resource management budgeting scheduling risk mitigation status reporting stakeholder communication pmp prince2',
        'senior project manager experienced in managing large scale it projects agile scrum waterfall methodology stakeholder management risk assessment budget control team leadership jira confluence',
        'pmo analyst experienced in project tracking reporting governance frameworks resource allocation budget management risk register milestone tracking dashboards kpi monitoring project portfolio management',
        'project management office lead skilled in establishing pmo frameworks governance standards project methodology resource management capacity planning portfolio management reporting stakeholder communication',
        'project manager pmp certified experienced in it project management agile scrum kanban waterfall stakeholder management risk mitigation budget control team leadership jira ms project',
    ],
    'SAP Developer': [
        'sap developer experienced in abap sap hana fiori ui5 sap mm sd fi co module configuration functional technical consulting integration development debugging performance optimization',
        'sap consultant skilled in sap ecc s4hana abap programming fiori development module configuration mm sd pp fi co integration middleware pi po data migration cutover',
        'senior sap technical consultant experienced in abap object oriented programming sap hana cloud platform fiori ui5 odata services workflow development smart forms adobe forms',
        'sap functional consultant experienced in sap mm sd fi co module implementation configuration testing training support business process mapping gap analysis requirements gathering',
        'sap abap developer proficient in classical and object oriented abap sap hana sql script cds views fiori elements odata services badi enhancement implementation',
        'sap developer with expertise in s4hana abap fiori ui5 module configuration mm sd fi integration development debugging performance tuning data migration testing',
    ],
    'Testing': [
        'software tester experienced in manual testing functional testing regression testing smoke testing sanity testing test case design execution defect tracking jira test management',
        'qa engineer skilled in manual testing test planning test strategy test case development execution defect lifecycle management regression testing integration testing system testing acceptance testing',
        'quality assurance professional experienced in software testing methodologies test case design black box white box testing defect management jira quality metrics reporting agile testing',
        'manual test engineer proficient in functional testing regression testing user acceptance testing test documentation test case writing execution defect reporting jira testlink quality assurance',
        'software testing professional skilled in manual testing test planning execution defect management test documentation browser testing mobile testing api testing database testing agile',
        'qa analyst experienced in manual software testing test case design execution defect tracking regression testing smoke testing integration testing system testing user acceptance testing',
    ],
}


def generate_training_data():
    """
    Generate preprocessed training data from the sample dataset.

    Returns:
        Tuple of (texts: list, labels: list)
    """
    texts = []
    labels = []

    for category, samples in SAMPLE_DATA.items():
        for sample in samples:
            cleaned = text_preprocessor.preprocess(sample)
            texts.append(cleaned)
            labels.append(category)

    return texts, labels


def train_model(model_dir=None):
    """
    Train the resume classifier and save the model.

    Args:
        model_dir: Directory to save model files. Defaults to 'ml_models/'.

    Returns:
        Dictionary with training metrics and evaluation report.
    """
    if model_dir is None:
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'ml_models'
        )

    print('=' * 60)
    print('Resume Classifier — Training Pipeline')
    print('=' * 60)

    # Step 1: Generate training data
    print('\n[1/5] Generating training data...')
    texts, labels = generate_training_data()
    print(f'  Total samples: {len(texts)}')
    print(f'  Categories: {len(set(labels))}')

    # Step 2: Split into train/test
    print('\n[2/5] Splitting data (80/20)...')
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f'  Training samples: {len(X_train)}')
    print(f'  Testing samples: {len(X_test)}')

    # Step 3: Train the model
    print('\n[3/5] Training TF-IDF + SVM classifier...')
    metrics = ResumeClassifier.train_and_save(X_train, y_train, model_dir)
    print(f'  Training accuracy: {metrics["accuracy"]:.4f}')
    print(f'  TF-IDF features: {metrics["n_features"]}')

    # Step 4: Evaluate on test set
    print('\n[4/5] Evaluating on test set...')
    classifier = ResumeClassifier()
    classifier.load_model(model_dir)

    y_pred = [classifier.predict(text)[0] for text in X_test]
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f'  Test accuracy: {test_accuracy:.4f}')
    print(f'\n  Classification Report:\n{report}')

    # Step 5: Save evaluation metrics
    print('\n[5/5] Saving metrics...')
    eval_metrics = {
        'train_accuracy': float(metrics['accuracy']),
        'test_accuracy': float(test_accuracy),
        'n_samples': metrics['n_samples'],
        'n_features': metrics['n_features'],
        'n_categories': metrics['n_categories'],
    }

    metrics_path = os.path.join(model_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    print(f'  Metrics saved to {metrics_path}')
    print(f'\n{"=" * 60}')
    print(f'Training complete! Model saved to {model_dir}')
    print(f'{"=" * 60}')

    return eval_metrics


if __name__ == '__main__':
    train_model()
