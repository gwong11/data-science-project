import React, { Component, useEffect, useState } from "react";
import { Dimensions, StyleSheet, View, Text, ImageBackground, Animated, Button } from "react-native"; 
import { Input } from 'react-native-elements';
import { PinchGestureHandler, State } from "react-native-gesture-handler"
import * as DocumentPicker from 'expo-document-picker';

import CustomHeader from "./CustomHeader";

/*
      <ImageBackground source={require('../assets/NIMH-Logo.png')}
                style={{width: '100%', height: '100%'}}
                imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
      >
      */
const { width } = Dimensions.get('window')

        /*<Text style={{fontSize: 26, fontWeight: 'bold', flex: 1}}>Settings page</Text>*/
const Settings = props => {
  
  const [vectorizer, setVectorizer] = useState("");
  const [vectorizerURI, setVectorizerURI] = useState("");
  const [vectorizerSize, setVectorizerSize] = useState(0);
  const [model, setModel] = useState("");
  const [modelURI, setModelURI] = useState("");
  const [modelSize, setModelSize] = useState(0);
  const [numberOfQueries, setNumberOfQueries] = useState(10);
  const [show, setShow] = useState(false);
  const [show2, setShow2] = useState(false);

  useEffect(() => {
      console.log(vectorizer);
      if (vectorizer != "") { 
         const postData = new FormData();
         postData.append("file", {
            uri: vectorizerURI,
            type: "application/octem-stream",
            name: vectorizer
         });
         fetch("http://192.168.1.152:8080/api/v1/setting", {
         //fetch("http://10.6.16.234:8080/api/v1/setting", {
             method: "POST",
             body: postData})
             .then(response => response.json())
             .then(data => {
                if (data.Reason) {
                   alert(data.Reason);
                }
                else {
                   //console.log(data);
                   //console.log(Object.keys(data).length);
                   console.log(data);
                   alert(data.message);
                   setShow(true);
                }
             })
             .catch((error) => {
                console.log(error);
             });
      }
  }, [vectorizer]);

  useEffect(() => {
      console.log(model);
      if (model != "") { 
         const postData = new FormData();
         postData.append("file", {
            uri: modelURI,
            type: "application/octem-stream",
            name: model
         });
         fetch("http://192.168.1.152:8080/api/v1/setting", {
         //fetch("http://10.6.16.234:8080/api/v1/setting", {
             method: "POST",
             body: postData})
             .then(response => response.json())
             .then(data => {
                if (data.Reason) {
                   alert(data.Reason);
                }
                else {
                   //console.log(data);
                   //console.log(Object.keys(data).length);
                   setShow2(true);
                }
             })
             .catch((error) => {
                console.log(error);
             });
      }
  }, [model]);

  const _pickVectorizer = async() => {
        let result = await DocumentPicker.getDocumentAsync({});
        //alert(result.uri);
        console.log(result);

        if (result.type == "success") {
            setVectorizer(result.name);
            setVectorizerURI(result.uri);
            setVectorizerSize(result.size);
            setShow(true);
        }
        else {
            setShow(false);
            alert("Please select vectorizer to upload!")
        }
  }

  const _pickModel = async () => {
        let result = await DocumentPicker.getDocumentAsync({});
        //alert(result.uri);
        console.log(result);

        if (result.type == "success") {
            setModel(result.name);
            setModelURI(result.uri);
            setModelSize(result.size);
            setShow2(true);
        }
        else {
            setShow2(false);
            alert("Please select model to upload!")
        }
    }

  const _setVectorizer = () => {
      console.log("Vect");
  }

  const _setModel = () => {
      console.log("Mod");
  }

  const _setNumberOfQueries = () => {
      console.log('Queries: ', numberOfQueries);
        if (numberOfQueries != 10) {
            fetch("http://192.168.1.152:8080/api/v1/setting", {
                method: "POST",
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({type: 'numberOfQueries', value: numberOfQueries})})
                .then(response => response.json())
                .then(data => {
                    console.log(numberOfQueries)
                })
                .catch((error) => {
                    setNumberOfQueries(10);
                    console.log(error);
                });
        }
  }

  return (
    <View style={styles.background}>
        <ImageBackground
              style={{width: '100%', height: '100%'}}
              imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
        >
            <CustomHeader navigation={props.navigation} title="Settings" />
            <View style={styles.container}>
                <View style={styles.inputContainer}>
                    <Button style={styles.input}
                        title="Upload Vectorizer"
                        color="blue"
                        onPress={_pickVectorizer}/>
                    {show ? (
                        <Input  editable={ false }
                                leftIcon={{ type: 'font-awesome', name: 'file' }}
                                value={ vectorizer }
                                multiline={true}
                                leftIconContainerStyle = {{marginLeft:10, padding: 10}}
                         />
                     ) : null }
                </View>
                <View style={styles.inputContainer}>
                    <Button style={styles.input}
                        title="Upload Model"
                        color="blue"
                        onPress={_pickModel}/>
                    {show2 ? (
                        <Input  editable={ false }
                                leftIcon={{ type: 'font-awesome', name: 'file' }}
                                value={ model }
                                multiline={true}
                                leftIconContainerStyle = {{marginLeft:10, padding: 10}}
                         />
                     ) : null }
                </View>
                <View style={styles.inputContainer}>
                   <Input  
                           editable={ true }
                           onChangeText={ numberOfQueries => {
                               setNumberOfQueries({ numberOfQueries })
                           }}
                    />
                    <Button style={styles.input}
                            title="Save"
                            color="green"
                            onPress={_setNumberOfQueries}/>
                </View>
            </View>
        </ImageBackground>
    </View>
  );
};

const styles = StyleSheet.create({
  background: {
    backgroundColor: '#add8e6',
    flex: 1
  },
  container: {
    flex: 1,
  },
  inputContainer: {
    alignItems: 'center',
    marginTop: 10,
    marginLeft: 20,
    padding: 25,
  },
  input: {
    margin: 15,
    width: width - 20,
    height: 50,
    borderColor: '#7a42f4',
    borderWidth: 1,
    marginLeft: 50,
    marginRight: 50,

   },
});

export default Settings;
