import React from "react";
import { Dimensions, StyleSheet, View, Text, ImageBackground, Animated } from "react-native";
import { PinchGestureHandler, State } from "react-native-gesture-handler"

import CustomHeader from "./CustomHeader";

/*
      <ImageBackground source={require('../assets/NIMH-Logo.png')}
                style={{width: '100%', height: '100%'}}
                imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
      >
      */
const { width } = Dimensions.get('window')

const Settings = props => {
  
  return (
    <View style={styles.background}>
        <ImageBackground source={require('../assets/NIMH-Logo.png')}
              style={{width: '100%', height: '100%'}}
              imageStyle={{resizeMode: 'contain', width: '50%', height: '180%', position : 'absolute', left: 200, right: 100}}
        >
            <CustomHeader navigation={props.navigation} title="Settings" />
            <View style={styles.container}>
                <Text style={{fontSize: 26, fontWeight: 'bold', flex: 1}}>Settings page</Text>
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
    marginLeft: 20
  },
});

export default Settings;
