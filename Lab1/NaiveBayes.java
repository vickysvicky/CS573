package NaiveBayes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/*
 * read in files
 * train NB classifier from training data
 * evaluate performance on training data
 * evaluate performance on testing data
 */
public class NaiveBayes {
	
	//data from file
	private static List<List<Integer>> train_data = new ArrayList<List<Integer>>();
	private static List<Integer> train_label = new ArrayList<Integer>();
	private static List<List<Integer>> test_data = new ArrayList<List<Integer>>();
	private static List<Integer> test_label = new ArrayList<Integer>();
	
	//constant 
	private static List<String> vocabulary = new ArrayList<String>();
	private static List<List<String>> mapping = new ArrayList<List<String>>();
	private static int v;
	
	//
	private static int n_cat;				//number of category
	private static int doc_total;			//number of document
	private static int[] doc_this;			//number of document in this category
	
	// NAIVE BAYES MODEL
	private static double[] Pwj; 			//CLASS PRIOR
	private static int[] cat_word; 			//TOTAL WORDS IN CAT, n
	private static int[][] vocab_cat; 		//OCCURENCE OF WORDS IN EACH CAT, nk
	private static double[][] PMLE; 		//MAX LIKELIHOOD ESTIMATOR
	private static double[][] PBE;			//BAYESIAN ESTIMATOR

	// DEBUG PURPOSES
	// set to 1 to print messages
	private static int debug = 0;
	
	public static void main(String args[]) {
		long initstart = System.currentTimeMillis();
		System.out.println("Reading in map and vocabulary...");
		// read in vocabulary
		File file_vocab = new File("vocabulary.txt");
		String word = "";
		try {
			Scanner sc = new Scanner(file_vocab);
			while(sc.hasNextLine()) {
				word = sc.nextLine();
				vocabulary.add(word);
			}
			sc.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		
		// read in map
		try(BufferedReader br = new BufferedReader(new FileReader("map.csv"));){
			String temp;
			while((temp=br.readLine())!=null) {
				String[] val = temp.split(",");
				mapping.add(Arrays.asList(val));
			}
			br.close();
		} catch (Throwable e) {
			e.printStackTrace();
		}
		n_cat = mapping.size();
		System.out.println("Number of category: "+n_cat);
		
		//if(args.length != 4) {
		//	System.out.println("You must provide paths to train_label train_data test_label test_data");
		//}else {

			String train_label_path = "train_label.csv"; //args[0];
			String train_data_path = "train_data.csv"; //args[1];
			String test_label_path = "test_label.csv"; //args[2];
			String test_data_path = "test_data.csv"; //args[3];
			System.out.println("Reading in data...");
			long starttime = System.currentTimeMillis();
			read_data(train_label_path, train_data_path, test_label_path, test_data_path);
			long endtime = System.currentTimeMillis();
			long elapse = (endtime - starttime);
			if(debug==1) {System.out.println("Elapsed time = "+elapse+"ms");}
			
			System.out.println("Training Naive Bayes model...");
			learn_NB();
			System.out.println("Class Prior:");
			double checkPwj = 0;
			for(int i=0; i<Pwj.length; i++) {
				int j=i+1;
				System.out.printf("P(Omega="+j+") = %.6f\n",Pwj[i]);
				checkPwj += Pwj[i];
			}
			if(debug==1) {System.out.println("total Pwj = "+checkPwj);}
		
			
			//EVALUATION OF TRAINING DATA
			System.out.println("Evaluation of training data:");
			starttime = System.currentTimeMillis();
			eval_NB(1);
			endtime = System.currentTimeMillis();
			elapse = (endtime - starttime);
			if(debug==1) {System.out.println("Elapsed time = "+elapse+"ms");}
			
			
			// EVALUATION OF TEST DATA
			System.out.println("Evaluation of test data:");
			starttime = System.currentTimeMillis();
			eval_NB(2);
			endtime = System.currentTimeMillis();
			elapse = (endtime - starttime);
			if(debug==1) {System.out.println("Elapsed time = "+elapse+"ms");}	
			
			long initend = System.currentTimeMillis();
			elapse = initend - initstart;
			System.out.println("Total time = "+elapse+"ms");
		//}
		
		
	}
	
	private static void read_data(String train_label_path, String train_data_path, String test_label_path, String test_data_path) {
		/*
		 * label doc: each row is a doc, val is the category
		 * data doc: each row is a word in a doc (docId, wordId, occurrence)
		 */
		
		// read in train label
		int i=0;
		try(BufferedReader br = new BufferedReader(new FileReader(train_label_path));){
			String temp;
			while((temp=br.readLine())!=null) {
				train_label.add(Integer.parseInt(temp));
				i++;
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e){
            System.out.println(e);
        }
		if(debug==1) {System.out.println("train label = "+i);}
		
		// read in test label
		i=0;
		try(BufferedReader br = new BufferedReader(new FileReader(test_label_path));){
			String temp;
			while((temp=br.readLine())!=null) {
				test_label.add(Integer.parseInt(temp));
				i++;
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e){
            System.out.println(e);
        }
		if(debug==1) {System.out.println("test label = "+i);}
		
		// read in train data
		i=0;
		try(BufferedReader br = new BufferedReader(new FileReader(train_data_path));){
			String temp;
			while((temp=br.readLine())!=null) {
				String[] val = temp.split(",");
				train_data.add(Arrays.asList(Integer.parseInt(val[0]),Integer.parseInt(val[1]),Integer.parseInt(val[2])));
				i++;
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e){
            System.out.println(e);
        }
		if(debug==1) {System.out.println("train data = "+i);}
		
		// read in test data
		i=0;
		try(BufferedReader br = new BufferedReader(new FileReader(test_data_path));){
			String temp;
			while((temp=br.readLine())!=null) {
				String[] val = temp.split(",");
				test_data.add(Arrays.asList(Integer.parseInt(val[0]),Integer.parseInt(val[1]),Integer.parseInt(val[2])));
				i++;
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e){
            System.out.println(e);
        }
		if(debug==1) {System.out.println("test data = "+i);}
	}	
	
	private static void learn_NB() {
		/*
		 * calculate
		 * - P(wj), class prior
		 * - n, total number of words in all doc in class wj
		 * - for each wk in V
		 * 		= nk, number of times wk occurs in all doc in class wj
		 * 		= P_MLE(wk|wj)=nk/n, max likelihood estimator
		 * 		= P_BE(wk|wj)=(nk + 1)/(n+|V|), Bayesian estimator/Laplace estimator
		 */
		
		doc_total = train_label.size();
		doc_this = new int[n_cat];
		v = vocabulary.size();
		
		// CLASS PRIOR & TOTAL WORDS IN CLASS & TOTA
		Pwj = new double[n_cat];
		cat_word = new int[n_cat];
		vocab_cat = new int[v][n_cat]; 
		//nk in wj is vocab_cat[k][j]
		
		//find number of doc in wj, doc_this
		for(int i=0; i<train_label.size(); i++) {
			doc_this[train_label.get(i)-1]++;
		}
		
		// for each wj
		for(int ii=0; ii<train_data.size(); ii++) { //for each word in doc
			int docID = train_data.get(ii).get(0)-1;
			int wj = train_label.get(docID)-1;
			cat_word[wj] += train_data.get(ii).get(2);
			vocab_cat[(train_data.get(ii).get(1))-1][wj] += train_data.get(ii).get(2);
		}
		
		
		for(int j=0; j<n_cat; j++) { 
			Pwj[j] = (double)doc_this[j]/(double)doc_total;
		}
		

		//P_MLE(wk|wj) is PMLE[k][j]
		PMLE = new double[v][n_cat];
		//P_BE(wj|wj) is PBE[k][j]
		PBE = new double[v][n_cat];
		// for each word k
		for(int k=0; k<v; k++) {
			for(int j=0; j<n_cat; j++) {
				 PMLE[k][j] = ((double)(vocab_cat[k][j]))/((double)(cat_word[j]));
				 PBE[k][j] = (double)(vocab_cat[k][j]+1)/(double)(cat_word[j]+v);
			}
		}
		
		for(int j=1; j<=n_cat; j++) {
			System.out.println("preview of P_MLE vs P_BE for word 1, 420, 6969 in category "+j);
			System.out.printf("P_MLE(w_k|omega_%d)\t P_BE(w_k|omega_%d)\n",j,j);
			System.out.printf("%.10f \t\t %.10f\n",PMLE[0][j-1],PBE[0][j-1]);
			System.out.printf("%.10f \t\t %.10f\n",PMLE[419][j-1],PBE[419][j-1]);
			System.out.printf("%.10f \t\t %.10f\n",PMLE[6968][j-1],PBE[6968][j-1]);
		}
		
		
		
	}
	
	private static void eval_NB(int d) {
		/*
		 * d=1 for train data
		 * d=2 for test data
		 * use NB to classify doc and check against label
		 */
		
		List<List<Integer>> data;
		List<Integer> label;
		int doc_num = 0;
		if(d==1) {
			data = train_data;
			label = train_label;
			doc_num = doc_total;
		}
		else {
			data = test_data;
			label = test_label;
			doc_num = test_label.size();
			
		}
		
		List<Integer> NB_label_MLE = new ArrayList<Integer>(); //wNB with MLE
		List<Integer> NB_label_BE = new ArrayList<Integer>(); //wNB with BE

		System.out.println("number of documents = "+ doc_num);
		
//		//find where each doc is at
//		int[] doc_index = new int[doc_num];
//		for(int i=0; i<data.size(); i++) {
//			doc_index[data.get(i).get(0)-1] = i;
//		}
		
		// CLASSIFY DOCUMENTS 2.0
		double[][] sumMLE = new double[doc_num][n_cat];
		double[][] sumBE = new double[doc_num][n_cat];
		double[][] MLE = new double[doc_num][n_cat];
		double[][] BE = new double[doc_num][n_cat];
		
		//find arg for all class of all doc
		for(int j=0; j<n_cat; j++) {
			// sum P(wk|wj)
			for(int ii=0; ii<data.size(); ii++) {
				int docID = data.get(ii).get(0)-1;
				int wordID = data.get(ii).get(1)-1;
				int nk = data.get(ii).get(2); 

				double logmle = Math.log(PMLE[wordID][j]);

				//if(!Double.isInfinite(logmle)){
					sumMLE[docID][j] += nk * logmle;
				//}
				//if(ii == 1000 && j == 15)
				//	System.out.println(sumMLE[docID][j]);
				sumBE[docID][j] += nk * Math.log(PBE[wordID][j]);
				
			}
			// multiply w P(wj)
			for(int i=0; i<doc_num; i++) {
				MLE[i][j] = Math.log(Pwj[j]) + sumMLE[i][j];
				BE[i][j] = Math.log(Pwj[j]) + sumBE[i][j];
			}
		}
		
		// find max of term and add the j to wnb
		
		for(int i=0; i<doc_num; i++) {
			int wNB_MLE = 0;
			int wNB_BE = 0;
			for(int j=1; j<n_cat; j++) {
				if(MLE[i][j]<MLE[i][wNB_MLE]) { //changed to find argmin?!?
					wNB_MLE = j;
				}
				if(BE[i][j]>BE[i][wNB_BE]) {
					wNB_BE = j;
				}
			}
			
			//label starts at 1 but index starts at 0
			wNB_MLE++;
			wNB_BE++;
			// add to NB_label

			NB_label_MLE.add(wNB_MLE);
			NB_label_BE.add(wNB_BE);
			if(debug==1) {
				System.out.println("label/MLE/BE: "+label.get(i)+"/"+wNB_MLE+"/"+wNB_BE);
			}  
		}
			
	
		
//		
//		// CLASSIFY DOCUMENTS
//		for(int i=0; i<doc_num; i++) {
//			double progress = ((double)i+1)/(double)doc_num *100.0;
//			if(debug==0) {System.out.printf("Progress: %.5f%% \r",progress);}
//			if(debug==1) {System.out.println("classifying doc "+i);}
//			
//			
//			// words in each wj in this doc
//			int doc_start = 0;
//			if(i!=0) {doc_start = doc_index[i-1]+1;}
//			int[] vocab_cat_data = new int[v]; 
//			for(int ii=doc_start; ii<=doc_index[i]; ii++) {
//				vocab_cat_data[(data.get(ii).get(1))-1]+=data.get(ii).get(2);
//			}
//			
//			
//			// find each term
//			double[] sum_MLE = new double[n_cat];
//			double[] sum_BE = new double[n_cat];
////			for(int j=0; j<n_cat; j++) {
////				double prod_MLE = 1;
////				double prod_BE = 1;
////				for(int k=0; k<v; k++) {
////					prod_MLE = prod_MLE*(Math.pow(PMLE[k][j],vocab_cat_data[k]));
////					prod_BE = prod_BE*(Math.pow(PBE[k][j],vocab_cat_data[k]));
////				}
////				sum_MLE[j] = Pwj[j]*prod_MLE;
////				sum_BE[j] = Pwj[j]*prod_BE;
////				if(debug==1) {System.out.printf("MLE=%.9f\t BE=%.9f",sum_MLE[j],sum_BE[j]);}
////			}
//			for(int j=0; j<n_cat; j++) {
//				double prod_MLE = 0;
//				double prod_BE = 0;
//				for(int k=0; k<v; k++) {
//					double logmle = Math.log(PMLE[k][j]);
//					if(!Double.isInfinite(logmle)){
//						prod_MLE += vocab_cat_data[k]*logmle;
//					}
//					prod_BE += vocab_cat_data[k]*Math.log(PBE[k][j]);
//				}
//				
//				sum_MLE[j] = Math.log(Pwj[j])+prod_MLE;
//				sum_BE[j] = Math.log(Pwj[j])+prod_BE;
//				//if(debug==1) {System.out.printf("MLE=%.9f\t BE=%.9f\n",sum_MLE[j],sum_BE[j]);}
//			}
//			
//			// find max of term and add the j to wnb
//			int wNB_MLE = 0;
//			int wNB_BE = 0;
//			for(int j=1; j<n_cat; j++) {
//				if(sum_MLE[j]<sum_MLE[wNB_MLE]) { //changed to find argmin?!?
//					wNB_MLE = j;
//				}
//				if(sum_BE[j]>sum_BE[wNB_BE]) {
//					wNB_BE = j;
//				}
//			}
//			//label starts at 1 but index starts at 0
//			wNB_MLE++;
//			wNB_BE++;
//			// add to NB_label
//
//			NB_label_MLE.add(wNB_MLE);
//			NB_label_BE.add(wNB_BE);	
//			if(debug==1) {
//				System.out.println("label/MLE/BE: "+label.get(i)+"/"+wNB_MLE+"/"+wNB_BE);
//			}
//		}
//		
		
		// sum of correctly classified document
		int sum_BE = 0;
		int sum_MLE = 0;
		int[] sum_class_BE = new int[n_cat];
		int[] sum_class_MLE = new int[n_cat];
		int[] doc_class = new int[n_cat];	//number of doc in class
		// CONFUSION MATRIX
		int[][] con_MLE = new int[n_cat][n_cat];
		int[][] con_BE = new int[n_cat][n_cat];
		for(int i=0; i<doc_num; i++) { 
			if(NB_label_BE.get(i).compareTo(label.get(i))==0) {
				sum_BE++;
				sum_class_BE[label.get(i)-1]++;
			}
			if(NB_label_MLE.get(i).compareTo(label.get(i))==0) {
				sum_MLE++;
				sum_class_MLE[label.get(i)-1]++;
			}
			doc_class[label.get(i)-1]++;
			con_MLE[label.get(i)-1][NB_label_MLE.get(i)-1]++;
			con_BE[label.get(i)-1][NB_label_BE.get(i)-1]++;
		}
		double acc_MLE = (double) sum_MLE/(double) doc_num;
		double acc_BE = (double) sum_BE/(double) doc_num;
		double[] acc_class_MLE = new double[n_cat];
		double[] acc_class_BE = new double[n_cat];
		
		System.out.printf("Overall accuracy (ML) = %.6f\n",acc_MLE);
		System.out.printf("Overall accuracy (Bayesian) = %.6f\n",acc_BE);
		System.out.println("Class Accuracy (MLE/BE):");
		for(int j=0; j<n_cat; j++) {
			acc_class_MLE[j] = (double) sum_class_MLE[j]/(double) doc_class[j];
			acc_class_BE[j] = (double) sum_class_BE[j]/(double) doc_class[j];
			System.out.printf("Group %d:\t%.6f\t%.6f\n",j+1,acc_class_MLE[j],acc_class_BE[j]);
		}
		
		System.out.println("Confusion Matrix (ML):");
		for(int j=0; j<n_cat; j++) {
			System.out.println(Arrays.toString(con_MLE[j]));
		}
		System.out.println("Confusion Matrix (Bayesian):");
		for(int j=0; j<n_cat; j++) {
			System.out.println(Arrays.toString(con_BE[j]));
		}	
	}
}












