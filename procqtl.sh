#Concat chromosomes
/mnt/tools/bcftools-1.16/bcftools concat /domino/datasets/local/Yazar_Imputed/QCed/chr{1..22}.dose.vcf.gz -Oz -o /mnt/results/eQTL/input/yazar.imputed.all.vcf.gz &

/mnt/tools/bcftools-1.16/bcftools concat /domino/datasets/local/Yazar_Imputed/allsamples/chr{1..22}.dose.vcf.gz -Oz -o /domino/datasets/local/Yazar_Imputed/allsamples/yazar.imputed.allsamples.vcf.gz &

#Create filtered vcf file for eQTL mapping
/mnt/tools/plink2 --vcf /domino/datasets/local/Yazar_Imputed/allsamples/yazar.imputed.allsamples.vcf.gz 'dosage=HDS' --keep /mnt/results/eQTL/input/yazar_ids_1104.iid --hwe 1e-6 --mac 20 --geno 0.05 --export vcf --out /mnt/results/eQTL/input/yazar.imputed.allsamples.filtered &


#zip and index vcf file
/mnt/tools/htslib-1.16/bgzip -c /mnt/results/eQTL/input/yazar.imputed.allsamples.filtered.vcf >/mnt/results/eQTL/input/yazar.imputed.allsamples.filtered.vcf.gz && /mnt/tools/htslib-1.16/tabix -p vcf /mnt/results/eQTL/input/yazar.imputed.allsamples.filtered.vcf.gz &

#zip and index bed file
/mnt/tools/htslib-1.16/bgzip /mnt/DISCOVER/eQTL/DISCOVER_WB_expr.wk0.bed && /mnt/tools/htslib-1.16/tabix -p bed /mnt/DISCOVER/eQTL/DISCOVER_WB_expr.wk0.bed.gz

/mnt/tools/htslib-1.16/bgzip/bgzip /mnt/DISCOVER/eQTL/DISCOVER_WB_expr.wk24.bed && /mnt/tools/htslib-1.16/bgzip/tabix -p bed /mnt/DISCOVER/eQTL/DISCOVER_WB_expr.wk24.bed.gz

#eQTL mapping
qtltool=$'//domino/datasets/GALAXI_Chr22_VCF/tools/qtltools/bin/QTLtools' 

##nominal version

files=$(ls /domino/datasets/local/Yazar_Imputed/eQTL/input/ | grep "expr_geneCount10pct"|grep "bed.gz$")

inpath="/domino/datasets/local/Yazar_Imputed/eQTL/input/"
outpath="/domino/datasets/local/Yazar_Imputed/eQTL/output/"

for file in $files
do
    name=$(echo $file|sed -e "s/.bed.gz//g")
    $qtltool cis --vcf /domino/datasets/local/Yazar_Imputed/eQTL/input/yazar.imputed.allsamples.filtered.vcf.gz --bed "${inpath}${file}" --cov "${inpath}${name}.cov"  --nominal 1.0 --normal --std-err --out "${outpath}${name}.txt" 
done




## Add header to all the files
files=$(ls /domino/datasets/local/Yazar_Imputed/eQTL/output/ | grep "expr_geneCount10pct"|grep ".txt$")
inpath="/domino/datasets/local/Yazar_Imputed/eQTL/output/"
outpath="/domino/datasets/local/Yazar_Imputed/eQTL/output/"

for file in $files
do
    name=$(echo $file|sed -e "s/.txt//g")
    cat "${input}header" "${inpput}${file}" > "${outpath}${name}_header.txt"
done


##Calculate aFC for qtltools outputs
afc=$'/mnt/tools/aFC/aFC.py'

$afc --vcf /mnt/DISCOVER/Genotype/allchr.imputed.DISCOVER.vcf.gz --pheno /mnt/DISCOVER/eQTL/WB/DISCOVER_WB_expr_overlap_genetics.wk0.bed.gz --qtl /mnt/DISCOVER/eQTL/WB/DISCOVER_WB_wk0_permutations_SVagesex.txt --geno 'GT' --chr 22 --log_xform 1 --log_base 2 --output /mnt/DISCOVER/eQTL/WB/acf_testing.txt
