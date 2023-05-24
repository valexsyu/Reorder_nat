#!/bin/sh
HOST=valexsyu@140.112.21.35
local_file=$1
destination_file=$2


#!/bin/bash

HOST="140.112.21.35"
USERNAME="valexsyu"
REMOTE_DIR=$1
LOCAL_DIR=$2
FILE=$3



#!/bin/bash

# Prompt for the SFTP password
read -s -p "Enter your SFTP password: " PASSWORD
echo

# Connect to the remote host and perform SFTP operations
echo "Connecting to $HOST..."
sftp "$USERNAME@$HOST" <<EOF
cd $REMOTE_DIR  # Change to the remote directory

# Download a file
get remote-file.txt $LOCAL_DIR/local-file.txt


# Close the SFTP session
quit
EOF

    