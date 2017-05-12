/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author PeDeNRiQue
 */
public class FileUtil {
    public static List<List<String>> readFile(String fileName,String separator){
        BufferedReader br = null;
        List<List<String>> matriz = new ArrayList<List<String>>();
        try {

                String sCurrentLine;

                br = new BufferedReader(
                        new FileReader(fileName));

                while ((sCurrentLine = br.readLine()) != null) {
                        //System.out.println(sCurrentLine);
                        String[] temp = sCurrentLine.split(separator);
                        List<String> linha = new ArrayList<String>();
                        for(String s : temp){
                            linha.add(s);
                        }
                        matriz.add(linha);
                }

//                for(int i = 0; i < matriz.size(); i++){
//                    for(int j = 0; j < matriz.get(i).size(); j++){
//                        System.out.println(matriz.get(i).get(j)+"<"+i+"|"+j+">");
//                    }
//                }

                System.out.println("Leu arquivo ");
        } catch (IOException e) {
                e.printStackTrace();
        } finally {
                try {
                        if (br != null)br.close();
                } catch (IOException ex) {
                        ex.printStackTrace();
                }
        }
        
        return matriz;
    }
    
    
    public static List<List<String>> readFile(String fileName,String separator,Integer totalRegistros){
        BufferedReader br = null;
        List<List<String>> matriz = new ArrayList<List<String>>();
        int cont = 0;
        try {

                String sCurrentLine;

                br = new BufferedReader(
                        new FileReader(fileName));

                while ((sCurrentLine = br.readLine()) != null && cont < totalRegistros) {
                        //System.out.println(sCurrentLine);
                        String[] temp = sCurrentLine.split(separator);
                        List<String> linha = new ArrayList<String>();
                        for(String s : temp){
                            //System.out.println("<>"+s+"<>");
                            linha.add(s);
                        }
                        matriz.add(linha);
                        cont++;
                }

                System.out.println("Leu arquivo ");
        } catch (IOException e) {
                e.printStackTrace();
        } finally {
                try {
                        if (br != null)br.close();
                } catch (IOException ex) {
                        ex.printStackTrace();
                }
        }
        
        return matriz;
    }
    
    public static List<List<String>> readAllFile(String fileName,String separator){
        BufferedReader br = null;
        List<List<String>> matriz = new ArrayList<List<String>>();
        int cont = 0;
        try {

                String sCurrentLine;

                br = new BufferedReader(new FileReader(fileName));

                while ((sCurrentLine = br.readLine()) != null) {
                        //System.out.println(sCurrentLine);
                        String[] temp = sCurrentLine.split(separator);
                        List<String> linha = new ArrayList<String>();
                        for(String s : temp){
                            //System.out.println("<>"+s+"<>");
                            linha.add(s);
                        }
                        matriz.add(linha);
                        cont++;
                }

                System.out.println("Leu arquivo ");
        } catch (IOException e) {
                e.printStackTrace();
        } finally {
                try {
                        if (br != null)br.close();
                } catch (IOException ex) {
                        ex.printStackTrace();
                }
        }
        
        return matriz;
    }
    
    public static void writeFile(String linha,String fileName){
        File file = new File(fileName);

        // if file doesnt exists, then create it
        FileWriter fw;
        BufferedWriter bw = null;
        
        try {
            if (!file.exists()) {
               file.createNewFile();
            }

            fw = new FileWriter(file.getAbsoluteFile(),true);
            bw = new BufferedWriter(fw);
            bw.write(linha+"\n");
        
        } catch (IOException ex) {
                Logger.getLogger(FileUtil.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            try {bw.close();} catch (Exception ex) {ex.printStackTrace();}
        }
    }
    
    public static String removeEspaco(String s){
        String nova = "";
        
        for(int i = 0; i < s.length(); i++){
          
             char c = s.charAt(i);  
             if(c == '.' || c == '0' || c == '1' || c == '2' || c == '3' || c == '4' || c == '5'
                     || c == '6' || c == '7' || c == '8' || c == '9'){
                 nova += c;
             }
        }
            
        return nova;
    }
    
    public static void showFile(List<List<String>> file){
        for(List<String> s : file){
            for(String ss : s){
                System.out.print(ss+" ");
            }
            System.out.println("");
        }
    }
}
