import com.google.gson.*;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class erroranalysis1 {
  public static void main(String[] args) throws Exception{
  
    //gliner errors
    int falsePositiveGLINER = 0; //found another wrong word and labelled it as correct
    int falseNegativeGLINER = 0; //didn't find the word at all
    int incorrectLabelGLINER = 0; //found the target word but labelled it something else 
    int boundaryErrorGLINER = 0; //didn't take in the entire complete word (only part)
    int totalErrorsGLINER = 0; //counter for total errors to do percents at the end and total

    //llama errors
    int falsePositiveLLAMA = 0; //found another wrong word and labelled it as correct
    int falseNegativeLLAMA = 0; //didn't find the word at all
    int incorrectLabelLLAMA = 0; //found the target word but labelled it something else 
    int boundaryErrorLLAMA = 0; //didn't take in the entire complete word (only part)
    int totalErrorsLLAMA = 0; //counter for total errors to do percents at the end and total

    //qwen errors
    int falsePositiveQWEN = 0; //found another wrong word and labelled it as correct
    int falseNegativeQWEN = 0; //didn't find the word at all
    int incorrectLabelQWEN = 0; //found the target word but labelled it something else 
    int boundaryErrorQWEN = 0; //didn't take in the entire complete word (only part)
    int totalErrorsQWEN = 0; //counter for total errors to do percents at the end and total

    //spacy errors
    int falsePositiveSPACY = 0; //found another wrong word and labelled it as correct
    int falseNegativeSPACY = 0; //didn't find the word at all
    int incorrectLabelSPACY = 0; //found the target word but labelled it something else 
    int boundaryErrorSPACY = 0; //didn't take in the entire complete word (only part)
    int totalErrorsSPACY = 0; //counter for total errors to do percents at the end and total

    //same code is reused for all of the json files different code at the very bottom to write the output of code
    JsonArray data = JsonParser.parseReader(
        //GLINER ERRORS
        new FileReader("error_examples_gliner_conll2003.json")
    ).getAsJsonArray();

      for (int i = 0; i < data.size(); i++) {

      JsonObject obj = data.get(i).getAsJsonObject();
      JsonArray missing = obj.getAsJsonArray("missing");
      JsonArray spurious = obj.getAsJsonArray("spurious");
      JsonArray gold = obj.getAsJsonArray("gold_entities");
      JsonArray pred = obj.getAsJsonArray("pred_entities");

        for (int j = 0; j < spurious.size(); j++) {
            
            JsonArray spuriousArr = spurious.get(j).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < missing.size(); k++) {

                JsonArray missingArr = missing.get(k).getAsJsonArray();

                int missingStart = missingArr.get(0).getAsInt();
                int missingEnd = missingArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {
                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }

            //if it isn't just mislabeled word, then its a false positive since it doesn't match anything
            if (incorrectLabel2) {
                incorrectLabelGLINER++;
                totalErrorsGLINER++;
            //if the ranges used are different naturally its a different word and we just check if its containinng each other earlieer
            }   else if (boundaryError2) {
                boundaryErrorGLINER++;
                totalErrorsGLINER++;
            }   else    {
                //we add false positive since it was in spurious but not in missing so we just made it up
                falsePositiveGLINER++;
                totalErrorsGLINER++;
            }
        }

        for (int j = 0; j < missing.size(); j++) {
            
            JsonArray missingArr = missing.get(j).getAsJsonArray();

            int missingStart = missingArr.get(0).getAsInt();
            int missingEnd = missingArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < spurious.size(); k++) {
            
            JsonArray spuriousArr = spurious.get(k).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {

                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }
                //the incorrect label is already counted on the other nested loop with spurious and missing so we do nothing
            if (incorrectLabel2) {}   
                //the boundary label is already counted on the other nested loop with spurious and missing so we do nothing
                else if (boundaryError2)   {    
                }   else    {
                //we add false negative since it was in missing but not in spurious so we weren't able to find it
                falseNegativeGLINER++;
                totalErrorsGLINER++;
            }
        }
    }

    JsonArray data2 = JsonParser.parseReader(
        new FileReader("error_examples_llama-3.1-8b_conll2003.json")
    ).getAsJsonArray();

      for (int i = 0; i < data2.size(); i++) {

      JsonObject obj = data2.get(i).getAsJsonObject();
      JsonArray missing = obj.getAsJsonArray("missing");
      JsonArray spurious = obj.getAsJsonArray("spurious");
      JsonArray gold = obj.getAsJsonArray("gold_entities");
      JsonArray pred = obj.getAsJsonArray("pred_entities");

        for (int j = 0; j < spurious.size(); j++) {
            
            JsonArray spuriousArr = spurious.get(j).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < missing.size(); k++) {

                JsonArray missingArr = missing.get(k).getAsJsonArray();

                int missingStart = missingArr.get(0).getAsInt();
                int missingEnd = missingArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {
                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }

            //if it isn't just mislabeled word, then its a false positive since it doesn't match anything
            if (incorrectLabel2) {
                incorrectLabelLLAMA++;
                totalErrorsLLAMA++;
            //if the ranges used are different naturally its a different word and we just check if its containinng each other earlieer
            }   else if (boundaryError2) {
                boundaryErrorLLAMA++;
                totalErrorsLLAMA++;
            }   else    {
                //we add false positive since it was in spurious but not in missing so we just made it up
                falsePositiveLLAMA++;
                totalErrorsLLAMA++;
            }
        }

        for (int j = 0; j < missing.size(); j++) {
            
            JsonArray missingArr = missing.get(j).getAsJsonArray();

            int missingStart = missingArr.get(0).getAsInt();
            int missingEnd = missingArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < spurious.size(); k++) {
            
            JsonArray spuriousArr = spurious.get(k).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {

                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }
                //the incorrect label is already counted on the other nested loop with spurious and missing so we do nothing
            if (incorrectLabel2) {}   
                //the boundary label is already counted on the other nested loop with spurious and missing so we do nothing
                else if (boundaryError2)   {    
                }   else    {
                //we add false negative since it was in missing but not in spurious so we weren't able to find it
                falseNegativeLLAMA++;
                totalErrorsLLAMA++;
            }
        }
    }

    JsonArray data3 = JsonParser.parseReader(
        new FileReader("error_examples_qwen-2.5-7b_conll2003.json")
    ).getAsJsonArray();

      for (int i = 0; i < data3.size(); i++) {

      JsonObject obj = data3.get(i).getAsJsonObject();
      JsonArray missing = obj.getAsJsonArray("missing");
      JsonArray spurious = obj.getAsJsonArray("spurious");
      JsonArray gold = obj.getAsJsonArray("gold_entities");
      JsonArray pred = obj.getAsJsonArray("pred_entities");

        for (int j = 0; j < spurious.size(); j++) {
            
            JsonArray spuriousArr = spurious.get(j).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < missing.size(); k++) {

                JsonArray missingArr = missing.get(k).getAsJsonArray();

                int missingStart = missingArr.get(0).getAsInt();
                int missingEnd = missingArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {
                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }

            //if it isn't just mislabeled word, then its a false positive since it doesn't match anything
            if (incorrectLabel2) {
                incorrectLabelQWEN++;
                totalErrorsQWEN++;
            //if the ranges used are different naturally its a different word and we just check if its containinng each other earlieer
            }   else if (boundaryError2) {
                boundaryErrorQWEN++;
                totalErrorsQWEN++;
            }   else    {
                //we add false positive since it was in spurious but not in missing so we just made it up
                falsePositiveQWEN++;
                totalErrorsQWEN++;
            }
        }

        for (int j = 0; j < missing.size(); j++) {
            
            JsonArray missingArr = missing.get(j).getAsJsonArray();

            int missingStart = missingArr.get(0).getAsInt();
            int missingEnd = missingArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < spurious.size(); k++) {
            
            JsonArray spuriousArr = spurious.get(k).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {

                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }
                //the incorrect label is already counted on the other nested loop with spurious and missing so we do nothing
            if (incorrectLabel2) {}   
                //the boundary label is already counted on the other nested loop with spurious and missing so we do nothing
                else if (boundaryError2)   {    
                }   else    {
                //we add false negative since it was in missing but not in spurious so we weren't able to find it
                falseNegativeQWEN++;
                totalErrorsQWEN++;
            }
        }
    }

    JsonArray data4 = JsonParser.parseReader(
        new FileReader("error_examples_spacy_conll2003.json")
    ).getAsJsonArray();

      for (int i = 0; i < data4.size(); i++) {

      JsonObject obj = data4.get(i).getAsJsonObject();
      JsonArray missing = obj.getAsJsonArray("missing");
      JsonArray spurious = obj.getAsJsonArray("spurious");
      JsonArray gold = obj.getAsJsonArray("gold_entities");
      JsonArray pred = obj.getAsJsonArray("pred_entities");

        for (int j = 0; j < spurious.size(); j++) {
            
            JsonArray spuriousArr = spurious.get(j).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < missing.size(); k++) {

                JsonArray missingArr = missing.get(k).getAsJsonArray();

                int missingStart = missingArr.get(0).getAsInt();
                int missingEnd = missingArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {
                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }

            //if it isn't just mislabeled word, then its a false positive since it doesn't match anything
            if (incorrectLabel2) {
                incorrectLabelSPACY++;
                totalErrorsSPACY++;
            //if the ranges used are different naturally its a different word and we just check if its containinng each other earlieer
            }   else if (boundaryError2) {
                boundaryErrorSPACY++;
                totalErrorsSPACY++;
            }   else    {
                //we add false positive since it was in spurious but not in missing so we just made it up
                falsePositiveSPACY++;
                totalErrorsSPACY++;
            }
        }

        for (int j = 0; j < missing.size(); j++) {
            
            JsonArray missingArr = missing.get(j).getAsJsonArray();

            int missingStart = missingArr.get(0).getAsInt();
            int missingEnd = missingArr.get(1).getAsInt();

            boolean incorrectLabel2 = false;
            boolean boundaryError2 = false;

            for (int k = 0; k < spurious.size(); k++) {
            
            JsonArray spuriousArr = spurious.get(k).getAsJsonArray();

            int spuriousStart = spuriousArr.get(0).getAsInt();
            int spuriousEnd = spuriousArr.get(1).getAsInt();

                //if the text retrieved is the same (same words retrieved and its both a missing and spurious error its just mislabelled)
                if (spuriousStart == missingStart && spuriousEnd == missingEnd) {
                    incorrectLabel2 = true;
                }   else    {

                    String goldWord = "";
                    String predWord = "";

                    //find word in gold
                    for (int l = 0; l < gold.size(); l++) {

                        JsonObject goldObj = gold.get(l).getAsJsonObject();

                        if (goldObj.get("start").getAsInt() == missingStart && goldObj.get("end").getAsInt() == missingEnd) {
                            goldWord = goldObj.get("text").getAsString();
                        }
                    }

                    //find word in pred
                    for (int m = 0; m < pred.size(); m++) {

                    JsonObject predObj = pred.get(m).getAsJsonObject();

                        if (predObj.get("start").getAsInt() == spuriousStart && predObj.get("end").getAsInt() == spuriousEnd) {
                            predWord = predObj.get("text").getAsString();
                        }
                    }

                    //if words are different and either gold word contains part of pred word or pred word contains part of gold word we're only getting part of the word so its a boundary error (also if not empty)
                    if (!goldWord.isEmpty() && !predWord.isEmpty() && !goldWord.equals(predWord) && (goldWord.contains(predWord) || predWord.contains(goldWord))) {
                    boundaryError2 = true;
                    }
                }
            }
                //the incorrect label is already counted on the other nested loop with spurious and missing so we do nothing
            if (incorrectLabel2) {}   
                //the boundary label is already counted on the other nested loop with spurious and missing so we do nothing
                else if (boundaryError2)   {    
                }   else    {
                //we add false negative since it was in missing but not in spurious so we weren't able to find it
                falseNegativeSPACY++;
                totalErrorsSPACY++;
            }
        }
    }

    try (BufferedWriter bw = new BufferedWriter(new FileWriter("phase1_error_analysis.txt"))) {
        bw.write("GLINER Error Analysis:");
        bw.newLine();
        bw.write("False Positives: " + falsePositiveGLINER + " (" + ((double) falsePositiveGLINER / totalErrorsGLINER) * 100 + "%)");
        bw.newLine();
        bw.write("False Negatives: " + falseNegativeGLINER + " (" + ((double) falseNegativeGLINER / totalErrorsGLINER) * 100 + "%)");
        bw.newLine();
        bw.write("Incorrect Labels: " + incorrectLabelGLINER + " (" + ((double) incorrectLabelGLINER / totalErrorsGLINER) * 100 + "%)");
        bw.newLine();
        bw.write("Boundary Errors: " + boundaryErrorGLINER + " (" + ((double) boundaryErrorGLINER / totalErrorsGLINER) * 100 + "%)");
        bw.newLine();
        bw.write("Total Errors: " + totalErrorsGLINER);
        bw.newLine();
        bw.newLine();

        bw.write("LLAMA-3.1-8B Error Analysis:");
        bw.newLine();
        bw.write("False Positives: " + falsePositiveLLAMA + " (" + ((double) falsePositiveLLAMA / totalErrorsLLAMA) * 100 + "%)");
        bw.newLine();
        bw.write("False Negatives: " + falseNegativeLLAMA + " (" + ((double) falseNegativeLLAMA / totalErrorsLLAMA) * 100 + "%)");
        bw.newLine();
        bw.write("Incorrect Labels: " + incorrectLabelLLAMA + " (" + ((double) incorrectLabelLLAMA / totalErrorsLLAMA) * 100 + "%)");
        bw.newLine();
        bw.write("Boundary Errors: " + boundaryErrorLLAMA + " (" + ((double) boundaryErrorLLAMA / totalErrorsLLAMA) * 100 + "%)");
        bw.newLine();
        bw.write("Total Errors: " + totalErrorsLLAMA);
        bw.newLine();
        bw.newLine();

        bw.write("QWEN-2.5-7B Error Analysis:");
        bw.newLine();
        bw.write("False Positives: " + falsePositiveQWEN + " (" + ((double) falsePositiveQWEN / totalErrorsQWEN) * 100 + "%)");
        bw.newLine();
        bw.write("False Negatives: " + falseNegativeQWEN + " (" + ((double) falseNegativeQWEN / totalErrorsQWEN) * 100 + "%)");
        bw.newLine();
        bw.write("Incorrect Labels: " + incorrectLabelQWEN + " (" + ((double) incorrectLabelQWEN / totalErrorsQWEN) * 100 + "%)");
        bw.newLine();
        bw.write("Boundary Errors: " + boundaryErrorQWEN + " (" + ((double) boundaryErrorQWEN / totalErrorsQWEN) * 100 + "%)");
        bw.newLine();
        bw.write("Total Errors: " + totalErrorsQWEN);
        bw.newLine();
        bw.newLine();

        bw.write("SPACY Error Analysis:");
        bw.newLine();
        bw.write("False Positives: " + falsePositiveSPACY + " (" + ((double) falsePositiveSPACY / totalErrorsSPACY) * 100 + "%)");
        bw.newLine();
        bw.write("False Negatives: " + falseNegativeSPACY + " (" + ((double) falseNegativeSPACY / totalErrorsSPACY) * 100 + "%)");
        bw.newLine();
        bw.write("Incorrect Labels: " + incorrectLabelSPACY + " (" + ((double) incorrectLabelSPACY / totalErrorsSPACY) * 100 + "%)");
        bw.newLine();
        bw.write("Boundary Errors: " + boundaryErrorSPACY + " (" + ((double) boundaryErrorSPACY / totalErrorsSPACY) * 100 + "%)");
        bw.newLine();
        bw.write("Total Errors: " + totalErrorsSPACY);
    } catch (IOException e) {
    }

  }
}