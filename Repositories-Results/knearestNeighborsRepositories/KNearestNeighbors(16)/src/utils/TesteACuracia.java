/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utils;

import java.util.List;

/**
 *
 * @author PeDeNRiQue
 */
public class TesteACuracia {
    public static void main(String[] args){
        String trainFileName = "submissao.txt";
        String separator = ",";
        
        List<List<String>> file = FileUtil.readFile(trainFileName, separator);
        int cont = 0;
        
        for(List<String> s : file){
            if(s.get(0).equals(s.get(1))){
                //System.out.println(s.get(0).equals(s.get(1)));
                cont++;
            }
        }
        System.out.println(cont/(file.size()*1.0));
    }
}
