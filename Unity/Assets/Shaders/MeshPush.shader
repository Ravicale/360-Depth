Shader "Custom/UnlitDisplacementMapping" {
	Properties {
		_Image ("Image", 2D) = "white" {}
		_HeightMap ("Depth", 2D) = "bump" {}  
		_Depth ("Depth", Range (0.0001, 1.0)) = 0.08
	}

	//Physically pushes vertices based on their normal using depthmap.
	//Slower than relief mapping, use only for debugging. Requires high resolution mesh.
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" "RenderType" = "Opaque"}
			CGPROGRAM
			#pragma vertex vertex_shader
			#pragma fragment pixel_shader
			#pragma target 4.0

			sampler2D _Image; 
			sampler2D _HeightMap;
			float _Depth;

			struct vertIn {
				float4 position : POSITION;
				float4 normal   : NORMAL;
				float2 texCoord : TEXCOORD0;
			};
			
			struct vertOut {
				float4 pos      : SV_POSITION;
				float2 texCoord : TEXCOORD0;
			};

			vertOut vertex_shader( vertIn vIn ) {
				vertOut vOut;
				vOut.pos = UnityObjectToClipPos(vIn.position + (vIn.normal * (_Depth * tex2Dlod(_HeightMap, float4(vIn.texCoord, 0.0, 0.0)))));
				vOut.texCoord = vIn.texCoord;

				return vOut;
			}

			float4 pixel_shader( vertOut fIn ): SV_TARGET {
				return tex2D(_Image, fIn.texCoord);
			}           
			ENDCG
		}
	}
}