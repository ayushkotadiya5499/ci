#!/bin/bash

# ૧. રીજન બદલીને us-east-1 કરો અને તમારો સાચો ECR URI વાપરો
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 982534348859.dkr.ecr.us-east-1.amazonaws.com

# ૨. તમારી સાચી ઇમેજ પુલ (Pull) કરો
docker pull 982534348859.dkr.ecr.us-east-1.amazonaws.com/salary_ecr:feature

# ૩. જો જૂનું કન્ટેનર ચાલતું હોય તો તેને બંધ કરો
if [ "$(docker ps -q -f name=salary-app)" ]; then
    docker stop salary-app
    docker rm salary-app
fi

# ૪. નવું કન્ટેનર રન કરો (પોર્ટ 80:8501 મુજબ)
# -p 80:8501 એટલે કે બહારથી 80 નંબરના પોર્ટ પર રિક્વેસ્ટ આવશે જે અંદર Streamlit ના 8501 પર જશે.
docker run -d -p 80:8501 --name salary-app 982534348859.dkr.ecr.us-east-1.amazonaws.com/salary_ecr:feature