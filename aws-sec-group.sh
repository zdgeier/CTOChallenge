aws ec2 describe-security-groups | jq ".SecurityGroups[] | .GroupName"
