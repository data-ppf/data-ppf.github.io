# get the data,
# extract only entries with complete birth+death after 999 and before 3000
# remove non-ascii characters with perl
# remove any spaces or "+" at beginning of line,
# dump into dat.asc

lynx \
    -width=999 \
    -dump \
    -hiddenlinks=ignore \
    -image_links=no \
    -minimal \
    -nobold \
    -nolist \
    -pseudo_inlines \
    -force_html \
  http://creativequotations.com//name-az.html \
  \
  | grep '(1[0-9][0-9][0-9]-[12][0-9][0-9][0-9])' \
  | /usr/bin/perl -pe 's/[^[:ascii:]]/+/g' \
  | sed -e 's/^[ +]*//' \
>! dat.asc

# strip out lines that only have 1 appearance of "-",
# i don't want to deal with "-" in your name or whatever in the 
# step when i change "-" to tab below
cat dat.asc \
  | awk -F\- 'NF==2 {print}' \
>! dat-v1.asc

# change (, ), or - to tabs
# note that this rests on there not being any "+" character
# anywhere in the file
cat dat-v1.asc \
  | sed -e 's/(/+/' -e 's/-/+/' -e 's/)/+/' \
  | tr '+' '\t' \
>! dat-v2.tsv

# now make a file with just age of death and job,
# and give it a rational name
cat dat-v2.tsv \
  | awk -F'\t' '{print $3-$2"\t"$4}'  \
  | sort -n \
  | grep -v '^0' \
>! tenure-job.tsv
# take the rationally-named tsv and create:

# stat on how many people have different jobs
cat tenure-job.tsv | cut -f2 |               sort -bfd | uniq -c | sort -n >! job-stats.txt
# stat on how many people have different primary jobs (before comma)
cat tenure-job.tsv | cut -f2 | cut -d, -f1 | sort -bfd | uniq -c | sort -n >! job-primary-stats.txt

# job-specific datafiles:
# poets, writers, singers
cat tenure-job.tsv | grep -i poet   | cut -f1 >!   tenure-poet.asc
cat tenure-job.tsv | grep -i writer | cut -f1 >! tenure-writer.asc
cat tenure-job.tsv | grep -i singer | cut -f1 >! tenure-singer.asc

cat dat-v2.tsv \
  | awk -F'\t' '{print $2,$3-$2}' \
  | sort -n \
  | grep ^1 \
>! born-tenure.asc
