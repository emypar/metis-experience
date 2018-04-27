#! /bin/bash --login

cd $(dirname $0) || exit 1
while [[ $# -gt 0 ]]; do
    aws_sync -aHovgS --exclude=/logs/ ./ $1:ds/metis/sf18_ds9/student_submissions/projects/kojak/parparita_emy
    shift
done
