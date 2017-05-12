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
public class Estatisticas {
    
    public static Double getMenorValor(List<List<String>> matriz, int nColuna){
        Double menor = null;
        Double d = null;
        
        for(int i = 0; i < matriz.size(); i++){
            if(nColuna < matriz.get(i).size() && isNumeric(matriz.get(i).get(nColuna))){
                
                d = Double.parseDouble(matriz.get(i).get(nColuna));
                //System.out.println(i+" "+d);
                if(menor ==  null){
                    menor = d;
                }else if(d < menor){
                    menor = d;
                }
            }                       
        }
        //System.out.println("Menor :"+menor);
        
        return menor;
    }
    
    public static Double getMaiorValor(List<List<String>> matriz, int nColuna){
        Double maior = null;
        Double d = null;
        
        for(int i = 0; i < matriz.size(); i++){
            if(nColuna < matriz.get(i).size() && isNumeric(matriz.get(i).get(nColuna))){
                
                d = Double.parseDouble(matriz.get(i).get(nColuna));
                //System.out.println(i+" "+d);
                if(maior ==  null){
                    maior = d;
                }else if(d > maior){
                    maior = d;
                }
            }                       
        }
        //System.out.println("Menor :"+menor);
        
        return maior;
    }
    
    public static boolean isNumeric(String str) {  
        try{ 
          Double d = Double.parseDouble(str);  
        }  
        catch(NumberFormatException nfe)  
        {  
          return false;  
        }  
        return true;  
    }
}
