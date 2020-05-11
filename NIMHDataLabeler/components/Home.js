import React, { Component, useEffect, useState } from 'react';
import { ScrollView, Alert, Dimensions, StyleSheet, View, Text, ImageBackground, TextInput, Button } from 'react-native';
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
    const [show, setShow] = useState(false);
    const [show2, setShow2] = useState(false);
    const [label, setLabel] = useState(null);
    const [test, setTest] = useState("");

    const _pickDocument = async () => {
	    let result = await DocumentPicker.getDocumentAsync({});
		//alert(result.uri);
        console.log(result);

        if (result.type == "success") {
            setFilename(result.name);
            setShow(true);
        }
        else {
            alert("Please select a data file to upload!")
            setShow(false);
        }
	}
    
    const _query = async () => {
        fetch("/")
            .then(response => response.json())
            .then(data => { 
                console.log(data);
                setTest(data); 
                setShow2(true)
            })
            .catch((error) => {
                console.log(error);
            });
    }

    const _handleLabel = (label) => {
        setLabel(label)
        console.log(label)
    }

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
                            <Button style={styles.button} title="Reset" color="red"/>
                        </View>
                        {show2 ? (
                            <View>
                                <View style={styles.outputContainer}>
                                    <Input editable={false}
                                        value=""
                                        multiline={true}
                                    />
                                </View>
                                <View style={styles.labelContainer}>
                                    <TextInput style={styles.labelInput}
                                               placeholder = "Label"
                                               placeholderTextColor = "#9a73ef"
                                               autoCapitalize = "none"
                                               onChangeText = {_handleLabel}
                                    />
                                </View>
                                <View style={styles.submitContainer}>
                                    <Button style={styles.button} 
                                        title="Submit" 
                                        color="grey"
                                    />
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
      fontSize: 20,
      justifyContent: 'space-between',
      padding: 10 
  },
  labelContainer: {
      marginLeft: 180
  },
  labelInput: {
      width: 50,
      height: 50,
      borderWidth: 1, 
      padding: 5,
  },
  submitContainer: {
      width: 100,
      marginTop: 50,
      marginLeft: 160,
      justifyContent: 'center'
  }
});

export default Home;
