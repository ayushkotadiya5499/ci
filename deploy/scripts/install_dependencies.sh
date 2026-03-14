#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y docker.io unzip curl

sudo systemctl start docker
sudo systemctl enable docker

# AWS CLI ઇન્સ્ટોલ કરવું જરૂરી છે જેથી ECR માંથી ઇમેજ ખેંચી શકાય
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install

sudo usermod -aG docker ubuntu
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws