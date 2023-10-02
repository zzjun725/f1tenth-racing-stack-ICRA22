#!/bin/bash
PACKAGE=(lane_follow opponent_predictor trajectory_generator dummy_car)
SUB_DIR=(config maps csv)

# if there are new output csv, put it under root csv folder
OUTPUTS=trajectory_generator/outputs
CSV=csv
if [ -d "./$OUTPUTS" ]; then
  # loop all folders
  for dir in "./$OUTPUTS"/*; do
    subdir="${dir##*/}"
    if [ -d "./$CSV/$subdir" ]; then
      rm -rf "./$CSV/$subdir/"
    fi
  done
  cp -r ./trajectory_generator/outputs/* ./csv/
fi

# loop all packages
for p in "${PACKAGE[@]}"; do
  # if package does not exist, move on
  if [ ! -d "./$p/" ]; then
    echo "Skipping $p package"
    continue
  fi

  # otherwise, build necessary subdirectories
  echo "Copying into $p package..."
  for s in "${SUB_DIR[@]}"; do
    if [ -d "./$p/$s/" ]; then
      rm -rf "./$p/$s/"
    fi
    mkdir -p "./$p/$s/"
    cp -r ./"$s"/* "./$p/$s/"
  done
done
