username=$1

declare -a servers=($2 $3 $4 $5 $6)

for server in "${servers[@]}"
do

ssh ${username}@${server} /bin/bash <<EOF
sudo apt-get update
sudo apt-get install -y screen python-pip python-dev vim
sudo pip install --upgrade pip
sudo pip install tensorflow
cp -r /proj/michigan-bigdata-PG0/assignments/assignment2/* .
EOF
scp -r "../EECS598-Assignment02" ${username}@${server}:~

done
