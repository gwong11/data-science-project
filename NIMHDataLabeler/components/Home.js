import React, { Component, useEffect, useState } from 'react';
import { KeyboardAvoidingView, Alert, Dimensions, StyleSheet, View, Text, ImageBackground, TextInput, Button } from 'react-native';
import { Input } from 'react-native-elements';
import ErrorBoundary from 'react-native-error-boundary';
import CustomHeader from "./CustomHeader";
import * as DocumentPicker from 'expo-document-picker';

var { height, width } = Dimensions.get('window');

const CustomFallback = (props: { error: Error, resetError: Function }) => {
  <View>
    <Text>Something happened!</Text>
    <Text>{props.error.toString()}</Text>
    <Button onPress={props.resetError} title={'Try again'} />
  </View>
}

//function Home(props) {
const Home = (props) => {
    
    const [filename, setFilename] = useState("");
    const [filenameURI, setFilenameURI] = useState("");
    const [fileSize, setFileSize] = useState(0);
    const [show, setShow] = useState(false);
    const [show2, setShow2] = useState(false);
    const [index, setIndex] = useState(0);
    const [jsonData, setJsonData] = useState("");
    const [label, setLabel] = useState(null);
    const [text, setText] = useState("");

    useEffect(() => {
        console.log('Index: ', index);
        let total = Object.keys(jsonData).length;
        if (index > 0 && index < total) {
            console.log(jsonData[index].text);
            setText(jsonData[index].text);
        }
        else {
            if (index != 0) {
                setFilename("");
                setFilenameURI("");
                setShow(false);
                setShow2(false);
                setLabel(null);
                setJsonData("");
                setIndex(0);
                setText("");
                alert("Training complete!");
            }
        }
    }, [index]);

    const _pickDocument = async () => {
	    let result = await DocumentPicker.getDocumentAsync({});
		//alert(result.uri);
        console.log(result);

        if (result.type == "success") {
            setFilename(result.name);
            setFilenameURI(result.uri);
            setFileSize(result.size);
            setShow(true);
        }
        else {
            alert("Please select a data file to upload!")
            setShow(false);
        }
	}

    const _query = async () => {
        if (filenameURI != "") {  
            setJsonData("");
            setText("");
            setShow2(false);
            const postData = new FormData();
            postData.append("file", {
                uri: filenameURI,
                type: "text/csv",
                name: filename
            });

            fetch("http://192.168.1.152:8080/api/v1/query", {
            //fetch("http://10.6.16.234:8080/api/v1/query", {
                method: "POST",
                body: postData})
                .then(response => response.json())
                .then(data => { 
                    if (data.Reason) {
                        alert(data.Reason);
                    }
                    else {
                        setJsonData(data);
                        //console.log(data);
                        //console.log(Object.keys(data).length);
                        setText(data[index].text); 
                        setShow2(true);
                    }
                })
                .catch((error) => {
                    console.log(error);
                });
        }
        else {
            alert("Something went wrong! Please make sure you upload a filename to begin!");
            setShow(false);
        }
    }

    const _reset = () => {
        setFilename("");
        setFilenameURI("");
        setShow(false);
        setShow2(false);
        setLabel(null);
        setJsonData("");
        setIndex(0);
        setText("");
        alert("Reset completed!");
    }

    const _handleLabel = (label) => {
        setLabel(label);
        console.log(label);
    }

    const _submit = () => {
        fetch("http://192.168.1.152:8080/api/v1/label", {
        //fetch("http://10.6.16.234:8080/api/v1/label", {
            method: "POST",
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({text: text, label: label})})
            .then(response => response.json())
            .then(data => { 
                if (data.code == "SUCCESS") {
                    console.log(data);
                    setIndex(index+1);
                }
                else if (data.code == "LABEL") {
                    setIndex(index+1);
                    alert(data.message);
                }
                else {
                    alert(data.message);
                }
             })
             .catch((error) => {
                 console.log(error);
             });
    }

    const _yes = () => {
        console.log("Yes");
    }

    const _no = () => {
        console.log("No");
    }

    const _quit = () => {
        console.log("Quit");
    }

             /*<Text style={{color: 'red', marginBottom: 40, fontSize: 15}}>Warning: Raises an error if record exists in database.</Text>*/
                                    /*<TextInput style={styles.labelInput}
                                               placeholder = "Label"
                                               placeholderTextColor = "#9a73ef"
                                               autoCapitalize = "none"
                                               onChangeText = {_handleLabel}
                                    />*/
    return (
      <View style={styles.background}>
        <ImageBackground
            style={{position: 'absolute', width: '100%', height: '100%'}}
            imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
        >
            <CustomHeader navigation={props.navigation} title="NIMH Data Labeler" />
            <View style={styles.container}>
                <ErrorBoundary FallbackComponent={CustomFallback}>
                        <View style={styles.inputContainer}>
                            <Button style={styles.input}
                                    title="Upload File" 
                                    onPress={_pickDocument}/>
                            {show ? (
                                <Input  editable={false} 
                                        leftIcon={{ type: 'font-awesome', name: 'file' }}
                                        value={ filename } 
                                        multiline={true}
                                        leftIconContainerStyle = {{marginLeft:10, padding: 10}}
                                />
                            ) : null }
                        </View>
                        <View style={styles.actionButtonContainer}>
                            <Button style={styles.button} 
                                    title="Query"
                                    color="green"
                                    onPress={_query}/>
                            <Button style={styles.button} 
                                    title="Reset" 
                                    color="red"
                                    onPress={_reset}/>
                        </View>
                        {show2 ? (
                            <View>
                                <View style={styles.outputContainer}>
                                    <Input inputStyle={{ fontSize: 14, color: "black", fontWeight: 'bold' }}
                                        editable={false}
                                        value={ text }
                                        multiline={true}
                                    />
                                </View>
                                <View style={styles.labelContainer}>
                                    <Button style={styles.button} 
                                        title="Yes"
                                        color="green"
                                        onPress={_yes}/>
                                    <Button style={styles.button} 
                                        title="No" 
                                        color="red"
                                        onPress={_no}/>
                                    <Button style={styles.button} 
                                        title="Quit" 
                                        color="red"
                                        onPress={_quit}/>
                                </View>
                            </View>
                        ) : null }
                </ErrorBoundary>
            </View>
        </ImageBackground>
    </View>
    )  
}

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
    marginTop: 50,
    padding: 20,
  },
  input: {
    margin: 15,
    width: width - 20,
    height: 50,
    borderColor: '#7a42f4',
    borderWidth: 1, 
    marginLeft: 50,
    marginRight: 50
   },
   actionButtonContainer: {
    width: '50%',
    marginLeft: 100,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  button: {
    flex: 1 
  },
  outputContainer: {
      marginLeft: 50,
      marginRight: 50,
      justifyContent: 'space-between',
      padding: 10,
  },
  labelContainer: {
      marginLeft: 180,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 20,
      marginLeft: 90,
      marginRight: 100
  },
  labelInput: {
      width: 50,
      height: 50,
      borderWidth: 1, 
      padding: 5,
  },
});

export default Home;
