import React, { Component } from 'react';
import { Alert, Dimensions, StyleSheet, View, Text, ImageBackground, TextInput, Button } from 'react-native';
import { Header } from 'react-native-elements';
import ErrorBoundary from 'react-native-error-boundary'
import CustomHeader from "./CustomHeader"

var { height, width } = Dimensions.get('window');

const CustomFallback = (props: { error: Error, resetError: Function }) => (
  <View>
    <Text>Something happened!</Text>
    <Text>{props.error.toString()}</Text>
    <Button onPress={props.resetError} title={'Try again'} />
  </View>
)

//export default function App() {
const Home = props => {
    return (
      <View style={styles.background}>
        <ImageBackground source={require('../assets/NIMH-Logo.png')}
            style={{width: '100%', height: '100%'}}
            imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
        >
            <CustomHeader navigation={props.navigation} title="NIMH Data Labeler" />
            <View style={styles.container}>
                <ErrorBoundary FallbackComponent={CustomFallback}>
                        <View style={styles.textInputContainer}>
                            <TextInput style = {styles.input}
                                underlineColorAndroid = "transparent"
                                placeholder = "Sentence Text"
                                placeholderTextColor = "#9a73ef"
                                autoCapitalize = "none"
                            />
                        </View>
                        <View style={styles.predictButtonContainer}>
                            <Button style={styles.button} 
                                    title="Query"
                                    color="green"
                                    onPress={() => Alert.alert('Button pressed')}/>
                        </View>
                        <View style={styles.resetButtonContainer}>
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
    justifyContent: 'space-around',
  },
  textInputContainer: {
    alignItems: 'center',
    marginTop: 250
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
  predictButtonContainer: {
    width: '70%',
    marginLeft: 80,
    flex: 1,
  },
  resetButtonContainer: {
    width: '70%',
    marginLeft: 80,
    flex: 1,
    marginTop: -310 
  },
  button: {
    backgroundColor: 'green',
    width: '40%',
    height: 40,
    flexDirection: 'row',
    flex: 1
  }
});

export default Home;
