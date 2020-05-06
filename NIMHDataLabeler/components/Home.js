import React, { Component, useEffect, useState } from 'react';
import { Alert, Dimensions, StyleSheet, View, Text, ImageBackground, TextInput, Button } from 'react-native';
import { DocumentPicker, ImagePicker } from 'expo';
import { Input } from 'react-native-elements';
import ErrorBoundary from 'react-native-error-boundary'
import CustomHeader from "./CustomHeader"

var { height, width } = Dimensions.get('window');

const CustomFallback = (props: { error: Error, resetError: Function }) => {
  <View>
    <Text>Something happened!</Text>
    <Text>{props.error.toString()}</Text>
    <Button onPress={props.resetError} title={'Try again'} />
  </View>
}

const TryUploadFile = () => {

    // Pick a single file
    try {
            const res = DocumentPicker.show({
            type: [DocumentPicker.types.allFiles],
            });
            console.log(
            res.uri,
            res.type, // mime type
            res.name,
            res.size
            );

    } catch (err) {
        if (DocumentPicker.isCancel(err)) {
        // User cancelled the picker, exit any dialogs or menus and move on
            alert('Canceled from document picker');
        } else {
            alert('Unknown Error: ' + JSON.stringify(err));
            throw err;
        }
    }
}

                            /*<TextInput style = {styles.input}
                                underlineColorAndroid = "transparent"
                                placeholder = "File Path"
                                placeholderTextColor = "#9a73ef"
                                autoCapitalize = "none"
                            />*/

//function Home(props) {
const Home = (props) => {
    
    const [bodyText, setBodyText] = useState("/home/G/sample.csv");
    const [show, setShow] = useState(false);

    /*useEffect(() => {
        fetch("/upload").then(response => 
            response.json().then(data => {
                setFilename(data); 
        })
      );
    }, "");*/


    const uploadFilename = () => {
        if (show == true) {
            setShow(false);
        } else {
            setShow(true);
        }
    }

    return (
      <View style={styles.background}>
        <ImageBackground source={require('../assets/NIMH-Logo.png')}
            style={{width: '100%', height: '100%'}}
            imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
        >
            <CustomHeader navigation={props.navigation} title="NIMH Data Labeler" />
            <View style={styles.container}>
                <ErrorBoundary FallbackComponent={CustomFallback}>
                        <View style={styles.inputContainer}>
                            <Button style={styles.input}
                                    title="Upload File" 
                                    onPress={uploadFilename}/>
                            {show ? (
                                <Input  editable={false} 
                                        leftIcon={{ type: 'font-awesome', name: 'file' }}
                                        value={ bodyText } 
                                        leftIconContainerStyle = {{marginLeft:10, padding: 10}}
                                />
                            ) : null }
                        </View>
                        <View style={styles.actionButtonContainer}>
                            <Button style={styles.button} 
                                    title="Query"
                                    color="green"
                                    onPress={() => Alert.alert('Button pressed')}/>
                            <Button style={styles.button} title="Reset" color="red"/>
                        </View>
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
  }
});

export default Home;
